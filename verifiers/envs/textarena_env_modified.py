import subprocess
from typing import List, Dict, Any, Optional, Tuple, Union
from copy import deepcopy
from verifiers import RewardFunc #type: ignore
from verifiers.envs.textarena_env import TextArenaEnv
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.textarena_rubric_modified import ModifiedTextArenaRubric
import textarena as ta #type: ignore
import time
from concurrent.futures import ThreadPoolExecutor
import random
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
import os
import json
from openai import OpenAI #type: ignore

class ModifiedTextArenaEnv(TextArenaEnv):
    def __init__(self,
                 system_prompt: Optional[str] = None,
                 few_shot: Optional[List[Dict[str, str]]] = None,
                 sampling_args: Dict[str, Any] = {
                     "stop": ["</response>"],
                 },
                 max_steps: int = 5,
                 env_id: str = "TruthAndDeception-v0", 
                 xml_parser: XMLParser = XMLParser(fields=["prediction", "think", "response"]),
                 answer_tag: str = "response",
                 prediction_tag: str = "prediction",
                 train_player_id: int = 0, **kwargs):
        super().__init__(
            system_prompt=system_prompt,
            few_shot=few_shot,
            sampling_args=sampling_args,
            train_player_id=train_player_id,
            env_id=env_id,
            xml_parser=xml_parser,
            answer_tag=answer_tag,
            max_steps=max_steps,
            **kwargs
        )
        self.env_player_id = 1 - self.train_player_id
        self.rubric = ModifiedTextArenaRubric(parser=self.parser)
        self.expert_agent = ta.agents.OpenAIAgent(model_name='gpt-4o-mini', use_reasoning_prompt=True)
        self.prediction_tag = prediction_tag
        self.env_answer_tag = "response"
        self.env_think_tag = "think"
        self.env_parser = XMLParser(fields=[self.env_answer_tag, self.env_think_tag])

    async def rollout(self,
                client: OpenAI,
                model: str,
                prompt: Messages,
                answer: str = "",
                task: str = "default",
                info: Info = {},
                sampling_args: SamplingArgs = {},
                **kwargs: Any) -> Tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        mode = kwargs.get("mode", "train")
        metadata = kwargs.get("metadata", None)
        is_completed = False
        state = {}
        assert isinstance(prompt, list), f"prompt : {prompt} is not a list"
        messages = deepcopy(prompt) 
        env = self.init_env_for_prompt(messages, mode=mode, metadata=metadata)
        game_prompt = env.generate_game_prompt_for_logging(player_id=self.train_player_id)
        state = {
            "env" : env,
            "player_id": self.train_player_id,
            "messages": messages,
            "game_prompt": game_prompt,
            "gt_opponent_thoughts": [],
            "predicted_opponent_thoughts": [],
            "messages_for_logging": [],
            "opponent_has_spoken": False,
            "responses": [],
        } 
        completion = []
        turn = 0
        if self.train_player_id == 1:
            env_response = self.do_initial_step_for_env(state)
            messages.append(env_response) #type: ignore
            completion.append(env_response)
        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                break
            response = await self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            state["responses"].append(response)
            assert isinstance(response, ChatCompletion), f"response : {response} is not a ChatCompletion"
            response_str = response.choices[0].message.content or ""
            self.simulate_train_player_step(response_str, state, turn=turn)
            messages.append({"role": "assistant", "content": response_str})
            completion.append({"role": "assistant", "content": response_str})
            if self.is_completed(messages, state, **kwargs) or (turn+1) >= self.max_turns:
                is_completed = True
            else:
                env_msg = self.simulate_env_player_step(state, turn=turn)
                messages.append(env_msg)
                completion.append(env_msg)
            turn += 1
        return completion, state

    def simulate_train_player_step(self, llm_response, state, turn: int): 
        parsed_text = self.parser.parse(llm_response)
        if hasattr(parsed_text, self.answer_tag):
            response = getattr(parsed_text, self.answer_tag)
            if response is None:
                split_response = llm_response.split(f"<{self.answer_tag}>")
                if len(split_response) > 1:
                    response = split_response[-1]
                else:
                    response = llm_response
        if hasattr(parsed_text, self.prediction_tag):
            prediction = getattr(parsed_text, self.prediction_tag)
            if prediction is None:
                split_response = llm_response.split("<think>")
                if len(split_response) > 1:
                    prediction = split_response[0]
                else:
                    prediction = llm_response
            if state['opponent_has_spoken']:    
                state['predicted_opponent_thoughts'].append(prediction)
        env = state['env']
        done, info = env.step(action=response)
        state["messages_for_logging"].append({
            "role": "assistant",
            "content": llm_response,
        })
        return done

    def do_initial_step_for_env(self, state: Dict[str, Any]) -> str :
        env_response = self.simulate_env_player_step(state, turn=0)
        last_message = state["messages"][-1]
        assert last_message["role"] == "user"
        state["messages"].append(env_response)
        return env_response #type: ignore
    
    def simulate_env_player_step(self, state: Dict[str, Any], turn: int) -> Dict[str, str]:  
        env = state["env"]
        player_id, obs = env.get_observation()
        assert player_id == self.env_player_id, f"player_id: {player_id}, env_player_id: {self.env_player_id}"
        raw_response, _ = self.expert_agent(obs)
        parsed_response = self.env_parser.parse(raw_response)
        if hasattr(parsed_response, self.env_answer_tag):
            action = getattr(parsed_response, self.env_answer_tag)
            if action is None:
                split_response = raw_response.split(f"<{self.env_answer_tag}>")
                if len(split_response) > 1:
                    action = split_response[-1]
                else:
                    action = raw_response
        if hasattr(parsed_response, self.env_think_tag):
            think = getattr(parsed_response, self.env_think_tag)
            if think is None:
                split_response = raw_response.split(f"<{self.env_answer_tag}>")
                if len(split_response) > 1:
                    think = split_response[0]
                else:
                    think = raw_response
            state['gt_opponent_thoughts'].append(think)
            state['opponent_has_spoken'] = True
        done, info = env.step(action)
        state["messages_for_logging"].append({
            "role": "user",
            "content": raw_response,
        })
        return {"role": "user", "content": action}
    
    def get_logging_data(self, all_prompts: List[Dict[str, Any]], all_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        updated_prompts = []
        updated_completions = []
        judge_logging_data = []
        
        for prompt, state in zip(all_prompts, all_states or []):
            game_prompt = state.get("game_prompt", "") # type: ignore
            if isinstance(prompt, str):
                updated_prompts.append(prompt + "\n\n**ONLY FOR LOGGING PURPOSES**\n\n" + game_prompt)
            elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[-1], dict):
                # Append game prompt to the last dict in the list
                updated_prompts.append(prompt[-1]['content'] + "\n\n**ONLY FOR LOGGING PURPOSES**\n\n"+ game_prompt)
            else:
                # If prompt is a list but not in chat format, just append the game prompt
                updated_prompts.append(game_prompt)

            messages_for_logging = state.get("messages_for_logging", []) # type: ignore
            updated_completions.append(messages_for_logging)
            
            # Retrieve judge logging data from state
            judge_data = state.get("judge_logging_data", "No judge logging data available") # type: ignore
            judge_logging_data.append(judge_data)
        return {"updated_prompts": updated_prompts, "updated_completions": updated_completions, "judge_logging_data": judge_logging_data}


        

    
    