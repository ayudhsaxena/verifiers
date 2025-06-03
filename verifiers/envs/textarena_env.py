import subprocess
from typing import List, Dict, Any

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import CODE_FEW_SHOT, CODE_PROMPT
from verifiers.rubrics import TextArenaRubric
from verifiers.utils import preprocess_dataset
import textarena as ta
import time
from concurrent.futures import ThreadPoolExecutor
from ..imports import LLM, SamplingParams  # type: ignore
import random
from typing import List, Dict, Sequence, Any, Union, Tuple
import os
import json

class TextArenaEnv(MultiStepEnv):
    def __init__(self,
                 dataset: str = "textarena",        
                 system_prompt: str = None,
                 few_shot: List[Dict[str, str]] = None,
                  sampling_args: Dict[str, Any] = {
                     "stop": ["</response>"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True, 
                 max_steps: int = 5,
                 env_id: str = "TruthAndDeception-v0", 
                 xml_parser: XMLParser = XMLParser(fields=["reasoning", "response"]),
                 answer_tag: str = "response",
                 train_player_id: int = 0, 
                 eval_dataset=None,**kwargs):
        super().__init__(
            system_prompt=system_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            sampling_args=sampling_args,
            **kwargs
        )
        self.train_player_id = train_player_id
        self.env_player_id = 1 - self.train_player_id
        self.env_id = env_id
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=system_prompt,
            few_shot=few_shot,
            env_id=self.env_id,
            player_id=self.train_player_id
        )
        self.eval_dataset = None
        if eval_dataset:
            self.eval_dataset = preprocess_dataset(
                dataset_name=eval_dataset,
                split="train",
                system_prompt=system_prompt,
                few_shot=few_shot,
            )

        self.max_steps = max_steps
        self.llm_parser = xml_parser
        self.env_parser = XMLParser(fields=["output"])
        self.answer_tag = answer_tag
        self.rubric = TextArenaRubric(parser=self.llm_parser, env_parser=self.env_parser)
        self.expert_agent = ta.agents.OpenAIAgent(model_name='gpt-4o-mini')

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        pass

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def get_eval_dataset(self, **kwargs: Any) -> Dataset:
        return self.eval_dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def is_completed(self, env: ta.Env) -> bool:
        done = env.is_completed()
        return done

        
    def init_env_for_prompt(self, message, mode = "train", **kwargs) -> ta.Env:
        prompt = message[0]['content'] if message[0]['role'] == 'user' else message[1]['content']
        metadata = kwargs.get("metadata", None)
        env = ta.make(env_id=self.env_id, mode=mode)
        env = ta.wrappers.LLMObservationWrapper(env=env)
        if self.env_id == "DontSayIt-v0":
            opponent_word = metadata.get("opponent_word", None)
            env.reset_with_prompt(prompt=prompt, opponent_word=opponent_word)
        else:
            env.reset_with_prompt(prompt=prompt)
        return env

    def simulate_env_player_step(self, state: Dict[str, Any]) -> Dict[str, str]:  
        # try:
            # if os.environ.get('LOCAL_RANK', '0') == '0' and not hasattr(self, '_breakpoint_called'):
            #     self._breakpoint_called = True
            #     breakpoint()
        env = state["env"]
        player_id, obs = env.get_observation()
        assert player_id == self.env_player_id
        raw_action, parsed_action = self.expert_agent(obs)
        action = parsed_action if parsed_action else raw_action
        done, info = env.step(action)
        return {"role": "user", "content": action}

    def get_reward_from_env(self, env: ta.Env, player_id: int = 0) -> float:
        return env.close()[player_id]

    def simulate_train_player_step(self, llm_responses, states, live_indices): 
        c = 0
        for i, j in enumerate(live_indices):
            state = states[j]
            text = llm_responses[i].outputs[0].text
            response = None
            parsed_text = self.llm_parser.parse(text)
            if hasattr(parsed_text, self.answer_tag):
                response = getattr(parsed_text, self.answer_tag)
            if response is None:
                
                print(f"Error: Could not parse response, defaulting to entire text")
                # # Create log_outputs directory if it doesn't exist
                # os.makedirs("log_outputs", exist_ok=True)
                # # Generate unique filename with timestamp
                # timestamp = time.strftime("%Y%m%d-%H%M%S")
                # filename = f"log_outputs/parse_error_{timestamp}_{i}.txt"
                # # Log the erroneous text
                # with open(filename, "w") as f:
                #     f.write(f"Failed to parse response:\n\n{text}")
                # print(f"Logged erroneous text to {filename}")
                response = text
            else:
                c += 1
            # print(f"Count of successful parses: {c}/{len(llm_responses)}");
            env = state['env']
            done, info = env.step(action=response)

        return done

    def do_initial_step_for_env(self, states: List[Dict[str, Any]]) :
        for state in states :
            env_response = self.simulate_env_player_step(state)
            last_message = state["messages"][-1]
            assert last_message["role"] == "user"
            state["messages"].append(env_response)


    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> List[Dict[str, Any]]:

        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]
        # for each list of messages, add an {role: 'assistant', content: '<reasoning>'} message
        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False) # type: ignore

        self.simulate_train_player_step(llm_responses, states, live_indices)

        def update_state(j, llm_response):
            # sleep for 0-1 seconds to avoid rate limiting
            time.sleep(self.sleep_time * random.random())
            state = states[j].copy()
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = llm_response.prompt_token_ids
            state["messages"].append({"role": "assistant", "content": llm_response.outputs[0].text})
        
            # get token lengths of env response and new completion
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len  = len(list(llm_response.prompt_token_ids)) - total_prev_len # type: ignore
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # update completion masks
            state["completion_mask"].extend([self.env_mask] * env_response_len)
            state["completion_mask"].extend([1] * new_completion_len)

            # update completion ids
            state["completion_ids"] = list(llm_response.prompt_token_ids) # type: ignore
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][len(state["prompt_ids"]):]
            
            if self.is_completed(state["env"]):
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                state["completion_mask"] = state["completion_mask"][:len(state["completion_ids"])]
            else:
                env_response = self.simulate_env_player_step(state)
                state["messages"].append(env_response)

            if not len(state["completion_mask"]) == len(state["completion_ids"]):
                print(f"Completion mask and completion ids are not the same length for state {j}")
                # Create log_error_state outputs directory if it doesn't exist
                os.makedirs("log_error_state", exist_ok=True)
                # Generate unique filename with timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"log_error_state/error_state_{timestamp}_{j}.json"
                # Save the state dict as json
                try:
                    # Create a serializable copy of the state
                    serializable_state = {}
                    for k, v in state.items():
                        if k == "env":
                            # Skip non-serializable environment object
                            continue
                        try:
                            # Test if value is JSON serializable
                            json.dumps({k: v})
                            serializable_state[k] = v
                        except (TypeError, OverflowError):
                            # If not serializable, convert to string representation
                            serializable_state[k] = str(v)
                    
                    with open(filename, "w") as f:
                        json.dump(serializable_state, f, indent=2)
                    print(f"Saved error state to {filename}")
                except Exception as e:
                    print(f"Failed to save error state: {e}")
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                min_len = min(len(state["completion_ids"]), len(state["completion_mask"]))
                state["completion_mask"] = state["completion_mask"][:min_len]
                state["completion_ids"] = state["completion_ids"][:min_len]
            return j, state
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda args: update_state(*args),
                [(j, llm_responses[i]) for i, j in enumerate(live_indices)]
            ))
       
        for j, state in results:
            if self.is_completed(state["env"]):
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                state["completion_mask"] = state["completion_mask"][:len(state["completion_ids"])]
            states[j] = state
        
        return states
    
    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] |  List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        mode = kwargs.get("mode", "train")
        metadata = kwargs.get("metadata", None)
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states = [{
            "env" : self.init_env_for_prompt(m, mode=mode, metadata=md),
            "messages": m,
            "prompt_messages": len(m),
            "prompt_ids": [],
            "completed": False,
            "completion_ids": [],
            "completion_mask": []
        } for m, md in zip(prompts, metadata)]
 
        if self.train_player_id == 1:
            self.do_initial_step_for_env(states)
        # main loop
        while not all_completed:
            states = self.step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)
        completion_messages = [s["messages"][s["prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        rewards = [self.get_reward_from_env(s["env"], player_id=self.train_player_id) if s["env"] else 0.0 for s in states]
        prompts_for_logging = [s["env"].generate_game_prompt_for_logging(player_id=self.train_player_id) if s["env"] else "" for s in states]
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
            "rewards": rewards,
            "prompts_for_logging": prompts_for_logging,
        }
        return output
        