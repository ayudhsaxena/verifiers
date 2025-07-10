from typing import List, Dict
from verifiers import RewardFunc
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from typing import List, Dict, Any


class TextArenaRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", "response"]),
                 env_parser: XMLParser = XMLParser(fields=["output"]),
                 funcs: List[RewardFunc] = [],
                 weights: List[float] = [],):
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.parser.get_format_reward_func(),
            self.game_termination_reward_func,
        ]
        self.reward_weights = [
            0.2,
            1.0,
        ]

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

   
    

