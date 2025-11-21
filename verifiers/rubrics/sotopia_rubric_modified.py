from typing import List, Dict, Any
import asyncio
import os
import weakref
import logging
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.judge_rubric import JudgeRubric


SOTOPIA_JUDGE_PROMPT = """The participant is engaging in a social interaction scenario and as part of their turn, they are trying to predict the other participant's thought process. We have the ground truth of the other participant's thoughts and the participant's prediction of the other participant's thoughts. Determine if the participant's prediction of the other participant's thoughts is correct.

Other participant's thoughts:
```
{thoughts}
```

Participant's Prediction of Other Participant's Thoughts:
```
{prediction}
```

Respond either "yes" or "no" only within the <answer></answer> tags. Then give a short explanation for your answer within the <reason></reason> tags.
Respond in the following format:
<answer>..</answer>
<reason>..</reason>
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
        ]
        self.reward_weights = [
            0.2,
            1.0,
        ]
        if include_judge_reward:
            self.reward_funcs.append(self.judge_reward_func)
            self.reward_weights.append(0.0)
        self.judge_prompt = SOTOPIA_JUDGE_PROMPT

        # Concurrency limit for judge API calls to avoid timeouts under heavy load
        # Can be tuned via env var JUDGE_CONCURRENCY
        self._judge_concurrency = int(os.getenv("JUDGE_CONCURRENCY", "32"))
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
        try:
            reward_sum = float(state.get("reward_sum", 0.0))/10.0
            self.logger.info(f"Reward sum: {reward_sum}")
            if reward_sum <= 0.7:
                return 0.0
            elif reward_sum < 0.9:
                return 0.5
            else:
                return 1.0
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