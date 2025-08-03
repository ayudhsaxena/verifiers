"""
ModifiedSotopiaEnv – a Multi-Turn environment that mirrors SotopiaEnv's workflow
but includes prediction tags for the train agent and think tags for the environment agent.
"""
import asyncio
from copy import deepcopy
from sympy import principal_branch
from torch.onnx.symbolic_opset9 import tanhshrink
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from openai import OpenAI                                   # LLM client used by verifiers
# External Sotopia imports – add `type: ignore` to silence static analysers when
# Sotopia is not installed in the current environment.
from sotopia.envs.parallel import ParallelSotopiaEnv  # type: ignore
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator, EpisodeLLMEvaluator, EvaluationForTwoAgents  # type: ignore
from sotopia.agents import LLMAgent, Agents  # type: ignore
from sotopia.messages import AgentAction, Observation  # type: ignore
from sotopia.database import AgentProfile, EnvironmentProfile  # type: ignore
from sotopia.database import SotopiaDimensions  # type: ignore

from verifiers.envs.sotopia_env import SotopiaEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import ModifiedSotopiaRubric
import random
import contextlib
import io
import json
try:
    from sotopia.generation_utils.output_parsers import PydanticOutputParser
except ImportError:
    # Fallback for when sotopia is not installed
    PydanticOutputParser = None


class ModifiedSotopiaEnv(SotopiaEnv):
    """
    Modified version of SotopiaEnv that includes prediction tags for the train agent
    and think tags for the environment agent, similar to ModifiedTextArenaEnv.
    """
    def __init__(
        self,
        # -- Sotopia specific ­-­
        train_player_id: int = 0,                                # 0 or 1
        evaluator: Optional[RuleBasedTerminatedEvaluator] = None,
        evaluator_model: str = "gpt-4o-mini",                    # Model for LLM-based evaluation
        # -- Verifiers plumbing ­-­
        system_prompt: Optional[str] = None,
        few_shot: Optional[List[Dict[str, str]]] = None,
        sampling_args: Dict[str, Any] = {},
        max_turns: int = 20,
        answer_tag: str = "response",
        think_tag: str = "think",
        prediction_tag: str = "prediction",
        suppress_output: bool = True,
        parser: XMLParser = XMLParser(fields=["prediction", "think", "response"]),
        **kwargs,
    ):
        super().__init__(
            train_player_id=train_player_id,
            evaluator=evaluator,
            evaluator_model=evaluator_model,
            system_prompt=system_prompt,
            few_shot=few_shot,
            sampling_args=sampling_args,
            max_turns=max_turns,
            answer_tag=answer_tag,
            think_tag=think_tag,
            suppress_output=suppress_output,
            parser=parser,
            **kwargs,
        )
        self.prediction_tag = prediction_tag
        self.env_answer_tag = "response"
        self.env_think_tag = "think"
        self.env_parser = XMLParser(fields=[self.env_answer_tag, self.env_think_tag])
        # Reward rubric specific to Modified Sotopia
        self.rubric = ModifiedSotopiaRubric(parser=parser)

    def _init_sotopia_env(self, prompt: List[Dict[str, Any]]) -> Tuple[ParallelSotopiaEnv, Dict[str, Any], LLMAgent, LLMAgent]:
        """
        Create a Sotopia simulator instance for a single dialogue.
        `prompt` is treated as the scenario string (same as SotopiaEnv's behaviour).
        """
        env_pk, agent1_pk, agent2_pk = prompt[-1]["content"].split(",")
        # Build ParallelSotopiaEnv
        env = ParallelSotopiaEnv(
            uuid_str=env_pk,
            action_order="round-robin",
            evaluators=self._evaluators,
            terminal_evaluators=self._terminal_evaluators,
        )

        train_agent = LLMAgent(model_name="dummy", uuid_str=agent1_pk)
        env_agent = LLMAgent(model_name="gpt-4o-mini", uuid_str=agent2_pk)
        
        agents = Agents(
            {
                train_agent.agent_name: train_agent,
                env_agent.agent_name: env_agent,
            }
        )

        if self.suppress_output:
            with contextlib.redirect_stdout(io.StringIO()):
                environment_messages = env.reset(
                    agents=agents,
                    omniscient=False,
                )
        else:
            environment_messages = env.reset(
                agents=agents,
                omniscient=False,
            )

        env_agent.reset()
        train_agent.reset()
        env_agent.goal = env.profile.agent_goals[self.env_player_id]
        train_agent.goal = env.profile.agent_goals[self.train_player_id]
        
        return env, environment_messages, train_agent, env_agent
    
    def _get_agent_action(self, agent: LLMAgent, env_obs: Observation, state: Dict[str, Any]) -> tuple[AgentAction, str]:
        """Run the agent's async action coroutine on the rollout-scoped event loop."""
        event_loop = state.get("event_loop")
        if event_loop is None:
            raise RuntimeError("Event loop missing from state – it should be created at rollout start.")

        return event_loop.run_until_complete(agent.aact(env_obs, use_prediction=True))



    def build_agent_action(self, raw_response: str, state: Dict[str, Any]) -> AgentAction:
        """Attempt to parse the LLM's output. If it is a JSON object with an
        ``argument`` field (as instructed by the template) we return that;
        otherwise we return the raw response unchanged."""

        try:
             # Parse prediction from the response
            parsed_text = self.parser.parse(raw_response)
            if hasattr(parsed_text, self.prediction_tag):
                prediction = getattr(parsed_text, self.prediction_tag)
                if prediction is None:
                    split_response = raw_response.split(f"<{self.think_tag}>")
                    if len(split_response) > 1:
                        prediction = split_response[0]
                    else:
                        prediction = raw_response
                if state['opponent_has_spoken']:    
                    state['predicted_opponent_thoughts'].append(prediction)
            if hasattr(parsed_text, self.answer_tag):
                response = getattr(parsed_text, self.answer_tag)
                if response is None:
                    split_response = raw_response.split(f"<{self.answer_tag}>")
                    if len(split_response) > 1:
                        response = split_response[-1]
                    else:
                        response = raw_response
            return self.output_parser.parse(response)
        except Exception as e:
            print(f"Error parsing agent action for string: {raw_response} with error: {e}")   
            return AgentAction(action_type="speak", argument=raw_response)

    def rollout(
        self,
        client: OpenAI,
        model: str,
        prompt: Union[str, List[Dict[str, Any]]],
        sampling_args: Dict[str, Any] = {},
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            assert isinstance(prompt, list), "Expect chat-formatted prompt"
            env, environment_messages, train_agent, env_agent = self._init_sotopia_env(prompt)
            prompt = prompt[:-1]  # remove the last message, which is the environment pk id

            state = {
                "env": env,
                "messages": deepcopy(prompt),
                "player_id": self.train_player_id,
                "reward_sum": 0.0,
                "terminated": False,
                "environment_messages": environment_messages,
                "train_agent": train_agent,
                "env_agent": env_agent,
                "env_background_prompt": env.background.to_natural_language(),
                "gt_opponent_thoughts": [],
                "predicted_opponent_thoughts": [],
                "opponent_has_spoken": False,
                "event_loop": loop,
            }
            completion: List[Dict[str, str]] = []
            turn = 0

            # If environment should start
            if self.train_player_id == 1:
                env_player_msg = self._simulate_env_player_step(state)
                completion.append(env_player_msg)
                state["messages"].append(env_player_msg)
            else:
                train_obs = state["environment_messages"][train_agent.agent_name]
                env_obs = state["environment_messages"][env_agent.agent_name]
                train_agent.update_inbox(train_obs)
                env_agent.update_inbox(env_obs)

            while not self.is_completed([], state) and turn < self.max_turns:
                # get train player response
                raw_assistant_response = self.get_model_response(
                    prompt=self.get_train_player_prompt(state),
                    client=client,
                    model=model,
                    sampling_args=sampling_args,
                    message_type=self.message_type,
                )

                assistant_action = self.build_agent_action(raw_assistant_response, state)
                self._simulate_train_player_step(assistant_action, state)

                completion.append({"role": "assistant", "content": raw_assistant_response})
                state["messages"].append({"role": "assistant", "content": raw_assistant_response})

                if self.is_completed([], state):
                    break

                env_player_msg = self._simulate_env_player_step(state)
                completion.append(env_player_msg)
                state["messages"].append(env_player_msg)

                turn += 1

            # pickling issue in <class 'sotopia.envs.evaluators.EvaluationForTwoAgents[SotopiaDimensions]'>
            if "env" in state:
                del state["env"]

            # Remove the loop reference from the state before returning
            if "event_loop" in state:
                del state["event_loop"]

            return completion, state

        finally:
            # Clean up: cancel any remaining tasks and close the loop.
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            loop.close()
            asyncio.set_event_loop(None)

    def _simulate_train_player_step(self, assistant_action: AgentAction, state: Dict[str, Any]) -> None:
        env: ParallelSotopiaEnv = state["env"]
        train_name = env.agents[self.train_player_id]
        env_name = env.agents[self.env_player_id]

        # environment player does nothing this sub-turn
        env_action = AgentAction(action_type="none", argument="")

        if self.suppress_output:
            with contextlib.redirect_stdout(io.StringIO()):
                (
                    environment_messages,
                    rewards,
                    terminated,
                    _,
                    info,
                ) = env.step({train_name: assistant_action, env_name: env_action})
        else:
            (environment_messages, rewards, terminated, _, info) = env.step({train_name: assistant_action, env_name: env_action})

        # Use only the 'goal' dimension as reward for the train agent
        complete_rating = info[train_name].get('complete_rating', 0)
        goal_score = 0
        if isinstance(complete_rating, tuple) and isinstance(complete_rating[1], dict) and "goal" in complete_rating[1]:
            goal_score = complete_rating[1]["goal"] 
        state["reward_sum"] += goal_score

        state["terminated"] = state["terminated"] or all(terminated.values())
        state["environment_messages"] = environment_messages

        

    def _simulate_env_player_step(self, state: Dict[str, Any]) -> Dict[str, str]:
        env: ParallelSotopiaEnv = state["env"]
        env_name = env.agents[self.env_player_id]
        train_name = env.agents[self.train_player_id]
        env_agent = state["env_agent"]
        train_agent = state["train_agent"]

        env_obs = state["environment_messages"][env_name]
        train_obs = state["environment_messages"][train_name]

        # manually update the train agent's inbox with its last observation
        train_agent.update_inbox(train_obs)

        # LLMAgent is async → block for simplicity
        env_action, think = self._get_agent_action(env_agent, env_obs, state)
        state['gt_opponent_thoughts'].append(think)
        state['opponent_has_spoken'] = True

        # now step with *only* env agent speaking
        if self.suppress_output:
            with contextlib.redirect_stdout(io.StringIO()):
                (
                    environment_messages,
                    rewards,
                    terminated,
                    _,
                    info,
                ) = env.step({train_name: AgentAction(action_type="none", argument=""),
                              env_name: env_action})
        else:
            (environment_messages, rewards, terminated, _, info) = env.step({train_name: AgentAction(action_type="none", argument=""), env_name: env_action})

        # Use only the 'goal' dimension as reward for the train agent
        complete_rating = info[train_name].get('complete_rating', 0)
        goal_score = 0
        if isinstance(complete_rating, tuple) and isinstance(complete_rating[1], dict) and "goal" in complete_rating[1]:
            goal_score = complete_rating[1]["goal"]
        state["reward_sum"] += goal_score

        state["terminated"] = state["terminated"] or all(terminated.values())
        state["environment_messages"] = environment_messages    

        train_obs = state["environment_messages"][train_name]
        env_obs = state["environment_messages"][env_name]
        train_agent.update_inbox(train_obs)
        env_agent.update_inbox(env_obs)


        raw_response = f"<{self.env_think_tag}>{think}</{self.env_think_tag}><{self.env_answer_tag}>{env_action.argument}</{self.env_answer_tag}>"
        # convert to chat message
        return {"role": "user", "content": raw_response}

    def get_logging_data(self, all_prompts: List[Union[str, Dict[str, Any], List[Dict[str, Any]]]], all_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        updated_prompts = []
        judge_logging_data = []
        
        for state in all_states:
            env_background_prompt = state.get("env_background_prompt", "") # type: ignore
            updated_prompts.append(env_background_prompt)
            # Retrieve judge logging data from state
            judge_data = state.get("judge_logging_data", "No judge logging data available") # type: ignore
            judge_logging_data.append(judge_data)
        
        return {"updated_prompts": updated_prompts, "judge_logging_data": judge_logging_data}

    def env_response(
        self,
        messages: List[Dict[str, Any]],
        state: Dict[str, Any],
        **kwargs: Any,
    ) :  # pragma: no cover
        pass

    def get_rubric(self, **kwargs: Any):
        """
        Return the rubric for this environment.
        """
        return self.rubric 