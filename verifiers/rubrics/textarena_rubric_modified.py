from typing import List, Dict
from verifiers import RewardFunc #type: ignore
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.judge_rubric import JudgeRubric
from typing import List, Dict, Any

TEXARENA_JUDGE_PROMPT = """The player is playing a game against the opponent and as part of its turn, its trying to predict the opponent's \
thought process. We have the ground truth of the opponent's thoughts and the player's prediction of the opponent's thoughts. \
Determine if the player's prediction of the opponent's thoughts is correct.

Opponent's thoughts:
```
{thoughts}
```

Player's Prediction of Opponent's Thoughts:
```
{prediction}
```

Respond either "yes" or "no" only within the <answer></answer> tags. Then give a short explanation for your answer within the <reason></reason> tags.
Respond in the following format:
<answer>..</answer>
<reason>..</reason>
"""


class ModifiedTextArenaRubric(JudgeRubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["prediction", "reasoning", "response"]),
                 env_parser: XMLParser = XMLParser(fields=["answer", "reason"]),
                 funcs: List[RewardFunc] = [],
                 weights: List[float] = [],):
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.parser.get_format_reward_func(),
            self.game_termination_reward_func,
            self.judge_reward_func
        ]
        self.reward_weights = [
            0.2,
            1.0,
            1.0,
        ]
        self.judge_prompt = TEXARENA_JUDGE_PROMPT

    def game_termination_reward_func(self, state: Dict[str, Any], **kwargs) -> float :
        # add exception handling for calculating rewards
        try:
            env = state["env"]
            player_id = state["player_id"]
            rewards = env.close()
            return rewards[player_id]
        except Exception as e:
            print(f"Error calculating game termination reward: {e}")
            return 0.0
        
    def judge_reward_func(self, state: Dict[str, Any], **kwargs) -> float:
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
        
        for i, (gt, pred) in enumerate(zip(gt_opponent_thoughts, predicted_opponent_thoughts)):
            if not isinstance(gt, str) or not isinstance(pred, str):
                raise ValueError("Ground truth and predicted opponent thoughts must be strings.")
            prompt = self.judge_prompt.format(thoughts=gt, prediction=pred)
            judge_response = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
            )
            judge_response = str(judge_response.choices[0].message.content)
            answer = self.env_parser.parse(judge_response).answer
            answer = answer if answer else judge_response
            is_correct = 'yes' in answer.lower()
            if is_correct:
                reward += 1.0
            
            # Create logging part for this comparison
            logging_part = f"Comparison {i+1}:\nGT Thought: {gt}\nPredicted Thought: {pred}\nJudge Response: {judge_response}\nCorrect: {is_correct}\n"
            logging_parts.append(logging_part)
        
        # Combine all logging parts and add final reward
        logging_string = "".join(logging_parts) + f"Final Judge Reward: {reward / len(gt_opponent_thoughts):.3f}"
        
        # Save the logging data in the state
        state["judge_logging_data"] = logging_string
        
        return reward / len(gt_opponent_thoughts) 
   
    

