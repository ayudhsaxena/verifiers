"""
SotopiaEnv ­–  a Multi-Turn environment that mirrors TextArenaEnv’s workflow
but drives a Sotopia social-scenario instead of a TextArena game.
"""
import asyncio
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import re

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
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.sotopia_rubric import SotopiaRubric
import random
import contextlib
import io
import json
from sotopia.generation_utils.output_parsers import  PydanticOutputParser
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
        # New: control which evaluation dimensions contribute to reward and which to log reasoning for
        reward_dimensions: Optional[List[str]] = None,
        reasoning_dimensions: Optional[List[str]] = None,
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
        # Default to goal-only, but allow flexible configuration
        self.reward_dimensions = reward_dimensions if reward_dimensions is not None else ["goal"]
        self.reasoning_dimensions = reasoning_dimensions if reasoning_dimensions is not None else list(self.reward_dimensions)

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
    
    async def _get_agent_action(self, agent: LLMAgent, env_obs: Observation, state: Dict[str, Any]) -> AgentAction:
        """Run the agent's async action coroutine on the rollout-scoped event loop."""
        return await agent.aact(env_obs)  # type: ignore[arg-type]

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

    def is_completed(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **_) -> bool:
        # termination handled by Sotopia evaluator
        return state.get("terminated", False)
    
    def get_train_player_prompt(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        env: ParallelSotopiaEnv = state["env"]
        train_name = env.agents[self.train_player_id]

        train_obs = state["environment_messages"][train_name]
        train_agent = state["train_agent"]
        prompt_content = train_agent.build_action_prompt(train_obs)

        train_prompt = []
        if self.system_prompt:
            train_prompt.append({"role": "system", "content": self.system_prompt})
        train_prompt.append({"role": "user", "content": prompt_content})
        return train_prompt

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
                assistant_action = self.build_agent_action(raw_assistant_response)
                await self._simulate_train_player_step(assistant_action, state)

                completion.append({"role": "assistant", "content": raw_assistant_response})
                state["messages"].append({"role": "assistant", "content": raw_assistant_response})

                if self.is_completed([], state):
                    break

                env_player_msg = await self._simulate_env_player_step(state)
                completion.append(env_player_msg)
                state["messages"].append(env_player_msg)

                turn += 1

            if "env" in state:
                del state["env"]

            return completion, state
        finally:
            pass

    async def _simulate_train_player_step(self, assistant_action: AgentAction, state: Dict[str, Any]) -> None:
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
                ) = await env.astep({train_name: assistant_action, env_name: env_action})
        else:
            (environment_messages, rewards, terminated, _, info) = await env.astep({train_name: assistant_action, env_name: env_action})

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

        env_action = await self._get_agent_action(env_agent, env_obs, state)

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
                              env_name: env_action})
        else:
            (environment_messages, rewards, terminated, _, info) = await env.astep({train_name: AgentAction(action_type="none", argument=""), env_name: env_action})

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
        comments = info[train_name].get("comments", None)
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

        # convert to chat message
        return {"role": "user", "content": env_action.argument}

    def get_logging_data(self, all_prompts: List[Dict[str, Any]], all_states: List[Dict[str, Any]]) -> Dict[str, Any]:
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

            # Merge into judge logging for visibility in existing logging pipeline
            if dim_reasoning:
                dims_label = ", ".join(
                    [f"{d} [{final_scores[d]:.3f}]" if isinstance(final_scores, dict) and d in final_scores else d for d in self.reasoning_dimensions]
                )
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