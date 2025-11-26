from typing import List, Dict, Any
import asyncio
import os
import weakref
import logging
import re
import json
from datetime import datetime
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.judge_rubric import JudgeRubric


SOTOPIA_JUDGE_PROMPT = """The participant is engaging in a social interaction scenario and as part of their turn, they are trying to predict the other participant's mental state. The mental state of a person is essentially what they believe, feel, want, desire, need, know etc. We have the ground truth of the other participant's mental state and the participant's prediction of the other participant's mental state. Determine if the participant's prediction of the other participant's mental state is correct.

Other participant's mental state:
```
{thoughts}
```

Participant's Prediction of Other Participant's mental state:
```
{prediction}
```

Respond either "yes" or "no" only within the <answer></answer> tags. Then give a short explanation for your answer within the <reason></reason> tags.
Respond in the following format:
<answer>..</answer>
<reason>..</reason>

NOTE: IF YOU FIND ANY KIND OF GIBBERISH IN THE TEXT LIKE REPEATED WORDS, NON-ENGLISH, NONSENSICAL PHRASES, JUST SIMPLY ASSIGN A SCORE OF 0.
                        
"""


class ModifiedSotopiaRubric(JudgeRubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["prediction", "think", "response"]),
                 judge_parser: XMLParser = XMLParser(fields=["answer", "reason"]),
                 funcs: List = [],
                 weights: List[float] = [],
                 include_judge_reward: bool = True,
                 **kwargs):
        super().__init__(funcs=funcs, weights=weights, parser=parser, **kwargs)
        self.parser = parser
        self.judge_parser = judge_parser
        # Build rewards conditionally to enable ablations between outcome (env) and process (judge) rewards
        self.reward_funcs = [
            self.parser.get_format_reward_func(),
            self.accumulated_env_reward_func,
            self.gibberish_reward_func,  # just for logging, not for reward
        ]
        self.reward_weights = [
            0.2,
            1.0,
            0.0,  # just for logging, not for reward
        ]
        if include_judge_reward:
            self.reward_funcs.append(self.judge_reward_func)
            self.reward_weights.append(0.0)
        self.judge_prompt = SOTOPIA_JUDGE_PROMPT

        # Concurrency limit for judge API calls to avoid timeouts under heavy load
        # Can be tuned via env var JUDGE_CONCURRENCY
        self._judge_concurrency = int(os.getenv("JUDGE_CONCURRENCY", "256"))
        # Per-event-loop semaphores to avoid cross-loop binding errors
        # Weakly reference loops so we don't prevent garbage collection
        self._judge_semaphores = weakref.WeakKeyDictionary()
        self.logger = logging.getLogger(__name__)

    def _get_judge_semaphore(self) -> asyncio.Semaphore:
        """Return a semaphore bound to the current running event loop.

        Avoids "Semaphore is bound to a different event loop" errors when this
        rubric is used from multiple threads/loops.
        """
        loop = asyncio.get_running_loop()
        semaphore = self._judge_semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self._judge_concurrency)
            self._judge_semaphores[loop] = semaphore
        return semaphore

    def accumulated_env_reward_func(self, state: Dict[str, Any], **_) -> float:  # noqa: D401
        """Return the sum of rewards collected for the training player.

        The rollout logic in ``SotopiaEnv`` keeps track of the cumulative reward
        in ``state["reward_sum"]``.
        """
        # Check for gibberish first - if detected, override reward to 0.0
        responses = state.get("responses", [])
        for response in responses:
            content = ""
            if hasattr(response, "choices"):
                content = response.choices[0].message.content or ""
            elif isinstance(response, dict):
                content = response.get("content", "")
            else:
                content = str(response)
                
            is_gibberish, reason = self._is_gibberish(content)
            if is_gibberish:
                self.logger.info(f"Gibberish detected in response: {content[:100]}... Reason: {reason}")
                
                # Save gibberish output to a file for analysis
                try:
                    gibberish_log_dir = "gibberish_logs"
                    os.makedirs(gibberish_log_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_file = os.path.join(gibberish_log_dir, f"gibberish_{timestamp}_{os.getpid()}.json")
                    
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "reason": reason,
                        "content": content,
                    }
                    
                    # Append to file if it exists, or create new (though timestamp in name makes unique files mostly)
                    # Using append mode with JSONL format (one JSON object per line) is usually cleaner for logs
                    with open(log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                        
                except Exception as e:
                    print(f"Failed to save gibberish log: {e}")

                return 0.0

        try:
            reward_sum = float(state.get("reward_sum", 0.0))
            self.logger.info(f"Reward sum: {reward_sum}")
            return reward_sum
            # return float(state.get("reward_sum", 0.0)) / 10.0
        except Exception as e:
            print(f"Error calculating accumulated environment reward: {e}")
            return 0.0
    
    async def judge_reward_func(self, state: Dict[str, Any], **kwargs) -> float:
        gt_opponent_thoughts = state.get("gt_opponent_thoughts", None)
        predicted_opponent_thoughts = state.get("predicted_opponent_thoughts", None)
        if gt_opponent_thoughts is None or predicted_opponent_thoughts is None:
            print("Ground truth opponent thoughts or predicted opponent thoughts are not provided.")
            # Create empty logging string
            state["judge_logging_data"] = "No ground truth or predicted thoughts available"
            return 0.0
        if len(gt_opponent_thoughts) != len(predicted_opponent_thoughts):
            assert len(gt_opponent_thoughts) == len(predicted_opponent_thoughts) + 1, \
                f"Ground truth opponent thoughts and predicted opponent thoughts should only differ by one if not equal.\n Ground truth: {len(gt_opponent_thoughts)} Predicted: {len(predicted_opponent_thoughts)}"
            gt_opponent_thoughts = gt_opponent_thoughts[:-1]

        reward = 0.0
        logging_parts = []
        step_rewards: list[float] = []
        
        for i, (gt, pred) in enumerate(zip(gt_opponent_thoughts, predicted_opponent_thoughts)):
            if not isinstance(gt, str) or not isinstance(pred, str):
                raise ValueError("Ground truth and predicted opponent thoughts must be strings.")
            prompt = self.judge_prompt.format(thoughts=gt, prediction=pred)
            # Limit concurrent judge calls to reduce timeouts under load
            async with self._get_judge_semaphore():
                judge_response = await self.judge_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    timeout=600,
                    **getattr(self, "judge_sampling_args", {}),
                )
            judge_response = judge_response.choices[0].message.content
            print(f"Judge Response: {judge_response}")
            answer = self.judge_parser.parse(judge_response).answer
            answer = answer if answer else judge_response
            is_correct = 'yes' in answer.lower()
            if is_correct:
                reward += 1.0
            step_rewards.append(1.0 if is_correct else 0.0)
            
            # Create logging part for this comparison
            logging_part = f"Comparison {i+1}:\nGT Thought: {gt}\nPredicted Thought: {pred}\nJudge Response: {judge_response}\nCorrect: {is_correct}\n"
            logging_parts.append(logging_part)
        
        # Combine all logging parts and add final reward
        logging_string = "".join(logging_parts) + f"Final Judge Reward: {reward / len(gt_opponent_thoughts):.3f}"
        
        # Save the logging data in the state
        state["judge_logging_data"] = logging_string
        # Save per-step rewards for process supervision consumers
        state["step_rewards"] = step_rewards
        
        return reward / len(gt_opponent_thoughts)

    def _is_gibberish(self, text: str) -> tuple[bool, str]:
        """Detect if text is gibberish using simple heuristics. Returns (is_gibberish, reason)."""
        if not text:
            return False, ""
        
        # 1. Check for specific malformed tag patterns
        # Matches "response>" etc not preceded by "<" or "/"
        tags = ["prediction", "think", "response"]
        for tag in tags:
             if re.search(rf"(?<![</])\b{tag}>", text, re.IGNORECASE):
                 return True, f"Malformed tag detected: {tag}>"
        
        # 2. Check for repeated words (3 or more times in a row)
        if re.search(r"\b(\w+)(?:\s+\1){2,}\b", text, re.IGNORECASE):
            return True, "Repeated words detected (3+ times)"

        # 3. Check for low unique word ratio
        words = re.findall(r"\w+", text)
        if len(words) > 30:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.15:
                return True, f"Low unique word ratio: {unique_ratio:.2f}"

        # 4. Check for malformed words using a dictionary
        try:
            import enchant
            d = enchant.Dict("en_US")
            
            # Filter for pure alphabetic words to avoid numbers/symbols
            alpha_words = [w for w in words if w.isalpha()]
            
            if len(alpha_words) > 10:
                # Count valid English words
                valid_word_count = sum(1 for w in alpha_words if d.check(w))
                valid_ratio = valid_word_count / len(alpha_words)
                
                # If less than 80% of words are valid English, likely gibberish
                if valid_ratio < 0.8:
                    return True, f"Low valid English word ratio: {valid_ratio:.2f}"
        except Exception as e:
            self.logger.warning(f"Enchant dictionary check failed: {e}")
        
        return False, ""

    def gibberish_reward_func(self, state: Dict[str, Any], **_) -> float:
        """
        Reward function that returns 0.0 if gibberish is detected, 1.0 otherwise.
        This can be used as a standalone reward component.
        """
        responses = state.get("responses", [])
        for response in responses:
            content = ""
            if hasattr(response, "choices"):
                content = response.choices[0].message.content or ""
            elif isinstance(response, dict):
                content = response.get("content", "")
            else:
                content = str(response)
                
            is_gibberish, reason = self._is_gibberish(content)
            if is_gibberish:
                # We already log in accumulated_env_reward_func, but if this is used standalone:
                # self.logger.info(f"Gibberish detected: {reason}")
                return 0.0
        
        return 1.0