"""
SotopiaEnv ­–  a Multi-Turn environment that mirrors TextArenaEnv’s workflow
but drives a Sotopia social-scenario instead of a TextArena game.
"""
import asyncio
from copy import deepcopy
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

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import SotopiaRubric
import random
import contextlib
import io
import json
from sotopia.generation_utils.output_parsers import  PydanticOutputParser


class SotopiaEnv(MultiTurnEnv):
    """
    Usage is identical to TextArenaEnv:
        env = SotopiaEnv(env_profile=some_profile_pk)
        completion, state = env.rollout(client, model, formatted_prompt)
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
        suppress_output: bool = True,
        parser: XMLParser = XMLParser(fields=["think", "response"]),
        **kwargs,
    ):
        super().__init__(
            system_prompt=system_prompt,
            few_shot=few_shot,
            sampling_args=sampling_args,
            max_turns=max_turns,
            **kwargs,
        )
        self.train_player_id = train_player_id
        self.env_player_id = 1 - self.train_player_id
        self.answer_tag = answer_tag
        self.think_tag = think_tag
        self.suppress_output = suppress_output
        self.parser = parser
        # Reward rubric specific to Sotopia
        self.rubric = SotopiaRubric(parser=parser)
        self.output_parser = PydanticOutputParser(pydantic_object=AgentAction)

        # Simple termination rule; plug-in any Sotopia evaluator(s)
        # Evaluators run every step; terminal evaluators run once at the end
        self._evaluators = [
            evaluator or RuleBasedTerminatedEvaluator(max_turn_number=max_turns)
        ]

        self._terminal_evaluators = [
            EpisodeLLMEvaluator(
                model_name=evaluator_model,
                response_format_class=EvaluationForTwoAgents[SotopiaDimensions]
            )
        ]


    def _init_sotopia_env(self, prompt: List[Dict[str, Any]]) -> ParallelSotopiaEnv:
        """
        Create a Sotopia simulator instance for a single dialogue.
        `prompt` is treated as the scenario string (same as TextArena’s behaviour).
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
    
    def _get_agent_action(self, agent: LLMAgent, env_obs: Observation, state: Dict[str, Any]) -> AgentAction:
        """Run the agent's async action coroutine on the rollout-scoped event loop."""
        loop = state.get("event_loop")
        if loop is None:
            raise RuntimeError("Event loop missing from state – it should be created at rollout start.")
        return loop.run_until_complete(agent.aact(env_obs))  # type: ignore[arg-type]

    def build_agent_action(self, raw_response: str) -> AgentAction:
        """Attempt to parse the LLM's output. If it is a JSON object with an
        ``argument`` field (as instructed by the template) we return that;
        otherwise we return the raw response unchanged."""

        try:
            parsed_response = self.parser.parse(raw_response).response
            if parsed_response is None:
                split_response = raw_response.split(f"<{self.answer_tag}>")
                if len(split_response) > 1:
                    parsed_response = split_response[-1]
                else:
                    parsed_response = raw_response  
            return self.output_parser.parse(parsed_response)
        except Exception as e:
            print(f"Error parsing agent action for string: {raw_response} with error: {e}")   
            return AgentAction(action_type="speak", argument=raw_response)

    def is_completed(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **_) -> bool:
        # termination handled by Sotopia evaluator
        return state.get("terminated", False)
    
    def get_train_player_prompt(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        env: ParallelSotopiaEnv = state["env"]
        train_name = env.agents[self.train_player_id]

        train_obs = state["environment_messages"][train_name]
        train_agent = state["train_agent"]
        prompt_content = train_agent.build_action_prompt(train_obs, use_prediction=True)

        train_prompt = []
        if self.system_prompt:
            train_prompt.append({"role": "system", "content": self.system_prompt})
        train_prompt.append({"role": "user", "content": prompt_content})
        return train_prompt

    def rollout(
        self,
        client: OpenAI,
        model: str,
        prompt: Union[str, List[Dict[str, Any]]],
        sampling_args: Dict[str, Any] = {},
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        # Create a dedicated event loop for this rollout/thread and set it as current.
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

                assistant_action = self.build_agent_action(raw_assistant_response)
                self._simulate_train_player_step(assistant_action, state)

                completion.append({"role": "assistant", "content": raw_assistant_response})
                state["messages"].append({"role": "assistant", "content": raw_assistant_response})

                if self.is_completed([], state):
                    break

                env_player_msg = self._simulate_env_player_step(state)
                completion.append(env_player_msg)
                state["messages"].append(env_player_msg)

                turn += 1

            if "env" in state:
                del state["env"]

            # Remove event loop reference before pickling/returning
            if "event_loop" in state:
                del state["event_loop"]

            return completion, state

        finally:
            # Cancel any remaining tasks and close the loop
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
        env_action = self._get_agent_action(env_agent, env_obs, state)

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

        # convert to chat message
        return {"role": "user", "content": env_action.argument}

    def get_logging_data(self, all_prompts: List[Dict[str, Any]], all_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        updated_prompts = []
        
        for state in all_states:
            env_background_prompt = state.get("env_background_prompt", "") # type: ignore
            updated_prompts.append(env_background_prompt)

        return {"updated_prompts": updated_prompts}

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