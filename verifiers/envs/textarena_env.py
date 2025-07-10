import subprocess
from typing import List, Dict, Any
from copy import deepcopy
from verifiers import RewardFunc #type: ignore
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import TextArenaRubric
import textarena as ta #type: ignore
import time
from concurrent.futures import ThreadPoolExecutor
import random
from typing import List, Dict, Sequence, Any, Union, Tuple, Optional
import os
import json
from openai import OpenAI

class TextArenaEnv(MultiTurnEnv):
    def __init__(self,
                 system_prompt: Optional[str] = None,
                 few_shot: Optional[List[Dict[str, str]]] = None,
                 sampling_args: Dict[str, Any] = {
                     "stop": ["</response>"],
                 },
                 max_steps: int = 5,
                 env_id: str = "TruthAndDeception-v0", 
                 xml_parser: XMLParser = XMLParser(fields=["reasoning", "response"]),
                 answer_tag: str = "response",
                 train_player_id: int = 0, **kwargs):
        super().__init__(
            system_prompt=system_prompt,
            few_shot=few_shot,
            sampling_args=sampling_args,
            **kwargs
        )
        self.train_player_id = train_player_id
        self.env_player_id = 1 - self.train_player_id
        self.env_id = env_id
        self.max_steps = max_steps
        self.parser = xml_parser
        self.answer_tag = answer_tag
        self.rubric = TextArenaRubric(parser=self.parser)
        self.expert_agent = ta.agents.OpenAIAgent(model_name='gpt-4o-mini')

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]: #type: ignore
        pass
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()
    
    def is_completed(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        env = state["env"]
        done = env.is_completed()
        return done
    
    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: Union[str, List[Dict[str, Any]]],
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        mode = kwargs.get("mode", "train")
        metadata = kwargs.get("metadata", None)
        is_completed = False
        state = {}
        assert isinstance(prompt, list)
        messages = deepcopy(prompt) 
        env = self.init_env_for_prompt(messages, mode=mode, metadata=metadata)
        game_prompt = env.generate_game_prompt_for_logging(player_id=self.train_player_id)
        state = {
            "env" : env,
            "player_id": self.train_player_id,
            "messages": messages,
            "game_prompt": game_prompt,
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
            response = self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            self.simulate_train_player_step(response, state)
            has_error = response.startswith("[ERROR]")
            messages.append({"role": "assistant", "content": response})
            completion.append({"role": "assistant", "content": response})
            turn += 1
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns or has_error:
                is_completed = True
            else:
                env_msg = self.simulate_env_player_step(state)
                messages.append(env_msg)
                completion.append(env_msg)
        return completion, state

        
    def init_env_for_prompt(self, message, mode = "train", **kwargs) -> ta.Env:
        prompt = message[0]['content'] if message[0]['role'] == 'user' else message[1]['content']
        metadata = kwargs.get("metadata", None)
        env = ta.make(env_id=self.env_id, mode=mode)
        env = ta.wrappers.LLMObservationWrapper(env=env)
        if self.env_id == "DontSayIt-v0":
            opponent_word = metadata.get("opponent_word", None) #type: ignore
            env.reset_with_prompt(prompt=prompt, opponent_word=opponent_word)
        else:
            env.reset_with_prompt(prompt=prompt)
        return env

    def simulate_env_player_step(self, state: Dict[str, Any]) -> Dict[str, str]:  
        env = state["env"]
        player_id, obs = env.get_observation()
        assert player_id == self.env_player_id
        raw_action, parsed_action = self.expert_agent(obs)
        action = parsed_action if parsed_action else raw_action
        done, info = env.step(action)
        return {"role": "user", "content": action}

    def get_reward_from_env(self, env: ta.Env, player_id: int = 0) -> float:
        return env.close()[player_id]

    def simulate_train_player_step(self, llm_response, state): 
        parsed_text = self.parser.parse(llm_response)
        if hasattr(parsed_text, self.answer_tag):
            response = getattr(parsed_text, self.answer_tag)
        if response is None:
            response = llm_response
        env = state['env']
        done, info = env.step(action=response)
        return done

    def do_initial_step_for_env(self, state: Dict[str, Any]) -> str :
        env_response = self.simulate_env_player_step(state)
        last_message = state["messages"][-1]
        assert last_message["role"] == "user"
        state["messages"].append(env_response)
        return env_response #type: ignore
    
    