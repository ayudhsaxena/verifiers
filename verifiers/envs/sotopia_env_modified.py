"""
ModifiedSotopiaEnv – a Multi-Turn environment that mirrors SotopiaEnv's workflow
but includes prediction tags for the train agent and think tags for the environment agent.
"""
import asyncio
from copy import deepcopy
from sympy import principal_branch
from torch.onnx.symbolic_opset9 import tanhshrink
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import re

from openai import OpenAI, AsyncOpenAI                                   # LLM client used by verifiers
# External Sotopia imports – add `type: ignore` to silence static analysers when
# Sotopia is not installed in the current environment.
from sotopia.envs.parallel import ParallelSotopiaEnv  # type: ignore
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator, EpisodeLLMEvaluator, EvaluationForTwoAgents  # type: ignore
from sotopia.agents import LLMAgent, Agents  # type: ignore
from sotopia.messages import AgentAction, Observation  # type: ignore
from sotopia.database import AgentProfile, EnvironmentProfile  # type: ignore
from sotopia.database import SotopiaDimensions  # type: ignore

from verifiers.envs.sotopia_env import SotopiaEnv
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.sotopia_rubric_modified import ModifiedSotopiaRubric
import random
import contextlib
import io
import json
try:
    from sotopia.generation_utils.output_parsers import PydanticOutputParser
except ImportError:
    # Fallback for when sotopia is not installed
    PydanticOutputParser = None
from sotopia.generation_utils.enums import MentalStateGeneration
from verifiers.types import (
    ChatCompletion,
    ChatMessage,
    Completion,
    Info,
    Messages,
    MessageType,
    SamplingArgs,
    State,
)

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
        environment_model: str = "gpt-4o-mini",  # Model for environment agent
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
        # New: control which evaluation dimensions contribute to reward and which to log reasoning for
        reward_dimensions: Optional[List[str]] = None,
        reasoning_dimensions: Optional[List[str]] = None,
        include_judge_reward: bool = True,
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
        self.env_parser = XMLParser(fields=[self.env_think_tag, self.env_answer_tag])
        self.environment_model = environment_model
        # Reward rubric specific to Modified Sotopia
        self.rubric = ModifiedSotopiaRubric(parser=parser, judge_client=AsyncOpenAI(), include_judge_reward=include_judge_reward)
        # Default to goal-only, but allow flexible configuration
        self.reward_dimensions = reward_dimensions if reward_dimensions is not None else ["goal"]
        self.reasoning_dimensions = reasoning_dimensions if reasoning_dimensions is not None else list(self.reward_dimensions)

    def _extract_dimensions_reasoning_from_comments(self, comments: str, train_player_id: int, target_dimensions: List[str]) -> str:
        """Extract reasoning lines for the specified dimensions for the training player.

        The evaluator comments string looks like:
        "Environment comments: ...\nAgent 1 comments:\n<dim>: <reason>\n...\nAgent 2 comments:\n...".
        We locate the appropriate agent section, then pull lines matching
        any of the provided `target_dimensions` (case-insensitive, exact key before ':').
        """
        if not isinstance(comments, str) or not comments:
            return ""

        target_agent_header = f"Agent {1 if train_player_id == 0 else 2} comments:"

        # Locate the start of the target agent section
        start_index = comments.find(target_agent_header)
        if start_index == -1:
            # Fallback: try to extract any matching lines across the entire comments
            collected: List[str] = []
            for dim in target_dimensions:
                pattern = rf"(?im)^\s*{re.escape(dim)}:\s*(.+)$"
                collected.extend(re.findall(pattern, comments))
            # If none are found, return empty string instead of full comments
            return " | ".join([line.strip() for line in collected]) if collected else ""

        section = comments[start_index + len(target_agent_header):]

        # End of section is the next agent header or end of string
        next_header_match = re.search(r"(?m)^Agent \s*[12]\s*comments:\s*$", section)
        agent_section = section[: next_header_match.start()] if next_header_match else section

        # Extract matching dimension lines within the agent section
        collected: List[str] = []
        for dim in target_dimensions:
            pattern = rf"(?im)^\s*{re.escape(dim)}:\s*(.+)$"
            collected.extend(f"{dim}: " + line.strip() for line in re.findall(pattern, agent_section))
        if collected:
            return " | ".join([line.strip() for line in collected])

        # If no matching lines are found within the agent section, return empty string
        return ""

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

        train_agent = LLMAgent(model_name="dummy", uuid_str=agent1_pk, mental_state_generation=MentalStateGeneration.FIRST_ORDER_MENTAL_STATE, mental_state_window=5)
        env_agent = LLMAgent(model_name=self.environment_model, uuid_str=agent2_pk, mental_state_generation=MentalStateGeneration.ZEROTH_ORDER_MENTAL_STATE, mental_state_window=5)
        
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
    
    async def _get_agent_action(self, agent: LLMAgent, env_obs: Observation) -> tuple[AgentAction, str]:
        """Run the agent's async action coroutine."""
        return await agent.aact(env_obs)



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

    async def rollout(
        self,
        client: OpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info = {},
        sampling_args: SamplingArgs = {},
        **kwargs,
    ) -> Tuple[Messages, State]:
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
                "responses": [],
                "dimension_reasoning_data": [],
                "final_dim_scores": {},
            }
            completion: List[Dict[str, str]] = []
            turn = 0

            # If environment should start
            if self.train_player_id == 1:
                env_player_msg = await self._simulate_env_player_step(state)
                completion.append(env_player_msg)
                state["messages"].append(env_player_msg)
            else:
                train_obs = state["environment_messages"][train_agent.agent_name]
                env_obs = state["environment_messages"][env_agent.agent_name]
                train_agent.update_inbox(train_obs)
                env_agent.update_inbox(env_obs)

            while not self.is_completed([], state) and turn < self.max_turns:
                # get train player response
                response = await self.get_model_response(
                    prompt=self.get_train_player_prompt(state),
                    client=client,
                    model=model,
                    sampling_args=sampling_args,
                    message_type=self.message_type,
                )
                state["responses"].append(response)
                assert isinstance(response, ChatCompletion), f"response : {response} is not a ChatCompletion"
                raw_assistant_response = response.choices[0].message.content or ""
                assistant_action = self.build_agent_action(raw_assistant_response, state)
                await self._simulate_train_player_step(assistant_action, state, raw_assistant_response)

                completion.append({"role": "assistant", "content": raw_assistant_response})
                state["messages"].append({"role": "assistant", "content": raw_assistant_response})

                if self.is_completed([], state):
                    break

                env_player_msg = await self._simulate_env_player_step(state)
                completion.append(env_player_msg)
                state["messages"].append(env_player_msg)

                turn += 1

            # pickling issue in <class 'sotopia.envs.evaluators.EvaluationForTwoAgents[SotopiaDimensions]'>
            if "env" in state:
                del state["env"]

            return completion, state

        finally:
            pass

    async def _simulate_train_player_step(self, assistant_action: AgentAction, state: Dict[str, Any], raw_assistant_response: str) -> None:
        env: ParallelSotopiaEnv = state["env"]
        train_name = env.agents[self.train_player_id]
        env_name = env.agents[self.env_player_id]

        # environment player does nothing this sub-turn
        env_action = AgentAction(action_type="none", argument="")

        # Prepare agent_messages_with_mental_state for observation creation
        agent_messages_with_mental_state = {
            train_name: raw_assistant_response,
            env_name: ""  # env agent is doing "none", so empty string
        }

        if self.suppress_output:
            with contextlib.redirect_stdout(io.StringIO()):
                (
                    environment_messages,
                    rewards,
                    terminated,
                    _,
                    info,
                ) = await env.astep({train_name: assistant_action, env_name: env_action}, agent_messages_with_mental_state=agent_messages_with_mental_state)
        else:
            (environment_messages, rewards, terminated, _, info) = await env.astep({train_name: assistant_action, env_name: env_action}, agent_messages_with_mental_state=agent_messages_with_mental_state)

        # Aggregate reward only from configured dimensions
        complete_rating = info[train_name].get('complete_rating', 0)
        step_reward = 0.0
        if isinstance(complete_rating, tuple) and isinstance(complete_rating[1], dict):
            dim_scores: Dict[str, Any] = complete_rating[1]
            # Persist the latest scores dict for end-of-episode logging
            state["final_dim_scores"] = {k: float(v) for k, v in dim_scores.items() if isinstance(v, (int, float))}
            for dim in self.reward_dimensions:
                if dim in dim_scores and isinstance(dim_scores[dim], (int, float)):
                    step_reward += float(dim_scores[dim])
        state["reward_sum"] += step_reward

        # Capture evaluator reasoning only for configured dimensions (if available)
        comments = info[train_name].get("comments", None)
        if comments:
            # Extract a combined reasoning string for configured dimensions
            dim_reason = self._extract_dimensions_reasoning_from_comments(
                comments, self.train_player_id, self.reasoning_dimensions
            )
            state["dimension_reasoning_data"].append(dim_reason)

        state["terminated"] = state["terminated"] or all(terminated.values())
        state["environment_messages"] = environment_messages

        

    async def _simulate_env_player_step(self, state: Dict[str, Any]) -> Dict[str, str]:
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
        env_action, env_raw_response = await self._get_agent_action(env_agent, env_obs)
        
        # Extract think from the raw response for logging
        parsed_text = self.env_parser.parse(env_raw_response)
        think = parsed_text.think if hasattr(parsed_text, 'think') and parsed_text.think is not None else ""
        if not think:
            # Fallback: try to extract think tag manually
            split_response = env_raw_response.split(f"<{self.env_answer_tag}>")
            if len(split_response) > 1:
                think = split_response[0]
            else:
                think = env_raw_response
        
        state['gt_opponent_thoughts'].append(think)
        state['opponent_has_spoken'] = True

        # Use the raw response directly for agent_messages_with_mental_state
        # The env agent uses ZEROTH_ORDER_MENTAL_STATE, so it has think and response tags
        agent_messages_with_mental_state = {
            train_name: "",  # train agent is doing "none", so empty string
            env_name: env_raw_response
        }

        # now step with *only* env agent speaking
        if self.suppress_output:
            with contextlib.redirect_stdout(io.StringIO()):
                (
                    environment_messages,
                    rewards,
                    terminated,
                    _,
                    info,
                ) = await env.astep({train_name: AgentAction(action_type="none", argument=""),
                              env_name: env_action}, agent_messages_with_mental_state=agent_messages_with_mental_state)
        else:
            (environment_messages, rewards, terminated, _, info) = await env.astep({train_name: AgentAction(action_type="none", argument=""), env_name: env_action}, agent_messages_with_mental_state=agent_messages_with_mental_state)

        # Aggregate reward only from configured dimensions
        complete_rating = info[train_name].get('complete_rating', 0)
        step_reward = 0.0
        if isinstance(complete_rating, tuple) and isinstance(complete_rating[1], dict):
            dim_scores: Dict[str, Any] = complete_rating[1]
            state["final_dim_scores"] = {k: float(v) for k, v in dim_scores.items() if isinstance(v, (int, float))}
            for dim in self.reward_dimensions:
                if dim in dim_scores and isinstance(dim_scores[dim], (int, float)):
                    step_reward += float(dim_scores[dim])
        state["reward_sum"] += step_reward

        # Capture evaluator reasoning only for configured dimensions (if available)
        comments = info[train_name].get("comments", None    )
        if comments:
            dim_reason = self._extract_dimensions_reasoning_from_comments(
                comments, self.train_player_id, self.reasoning_dimensions
            )
            state["dimension_reasoning_data"].append(dim_reason)

        state["terminated"] = state["terminated"] or all(terminated.values())
        state["environment_messages"] = environment_messages    

        train_obs = state["environment_messages"][train_name]
        env_obs = state["environment_messages"][env_name]
        train_agent.update_inbox(train_obs)
        env_agent.update_inbox(env_obs)

        # convert to chat message (reuse env_raw_response that was created earlier)
        return {"role": "user", "content": env_raw_response}

    def get_logging_data(self, all_prompts: List[Union[str, Dict[str, Any], List[Dict[str, Any]]]], all_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        updated_prompts: List[str] = []
        judge_logging_data: List[str] = []
        dimension_reasoning_logs: List[str] = []

        for state in all_states:
            env_background_prompt = state.get("env_background_prompt", "")  # type: ignore
            updated_prompts.append(env_background_prompt)

            # Retrieve judge logging data from state
            judge_data = state.get("judge_logging_data", "No judge logging data available")  # type: ignore

            # Retrieve accumulated reasoning strings for configured dimensions
            reasons = state.get("dimension_reasoning_data", [])  # type: ignore
            if isinstance(reasons, list):
                # Join per-turn reasons into a single string
                dim_reasoning = "\n".join([str(r) for r in reasons if r])
            else:
                dim_reasoning = str(reasons) if reasons else ""

            # Append scores inline to the flat reasoning text if final scores are available
            final_scores: Dict[str, float] = state.get("final_dim_scores", {})  # type: ignore
            if dim_reasoning and isinstance(final_scores, dict) and final_scores:
                scored_text = dim_reasoning
                for dim in self.reasoning_dimensions:
                    if dim in final_scores:
                        scored_text = re.sub(
                            rf"(?im)(^|\s){re.escape(dim)}\s*:",
                            lambda m, d=dim: f"{m.group(1)}{d} [{final_scores[d]:.3f}]:",
                            scored_text,
                        )
                dim_reasoning = scored_text

            # Save standalone reasoning for configured dimensions
            dimension_reasoning_logs.append(dim_reasoning)

            # Build per-episode flat string with appended scores using the final complete_rating if available
            # We source scores at episode end where they become non-null in the info tuple.
            final_scores: Dict[str, float] = {}
            # Best-effort: try to reconstruct from last available comments via evaluator, else leave blank
            # Scores will be appended later in the combined output when available in the trainer context.

            # Merge into judge logging for visibility in existing logging pipeline
            if dim_reasoning:
                dims_label = ", ".join(self.reasoning_dimensions)
                # Append scores from final_dim_scores if available
                scores_suffix = ""
                final_scores: Dict[str, float] = state.get("final_dim_scores", {})  # type: ignore
                if isinstance(final_scores, dict) and final_scores:
                    scored_dims = [
                        f"{dim} [{final_scores[dim]:.3f}]" if dim in final_scores else dim
                        for dim in self.reasoning_dimensions
                    ]
                    dims_label = ", ".join(scored_dims)
                combined = f"{judge_data}\n Final Reward reasoning [{dims_label}]:\n{dim_reasoning}"
            else:
                combined = judge_data
            judge_logging_data.append(combined)

        return {
            "updated_prompts": updated_prompts,
            "judge_logging_data": judge_logging_data,
            "dimension_reasoning_data": dimension_reasoning_logs,
        }

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