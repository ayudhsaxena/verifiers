from typing import List, Dict, Any

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric


class SotopiaRubric(Rubric):
    """Rubric for the Sotopia multi-turn social-interaction environment.

    Currently it implements two simple reward functions:
    1. ``format_reward_func`` – ensures the model follows the required XML format
       (delegated to ``XMLParser.get_format_reward_func``).
    2. ``accumulated_env_reward`` – returns the cumulative reward collected from
       the underlying Sotopia environment during the episode.  ``SotopiaEnv``
       stores this under ``state["reward_sum"]``.

    The weights are set such that environment rewards dominate (1.0) while the
    formatting reward provides a gentle incentive to keep the output valid
    (0.2).
    """

    def __init__(
        self,
        parser: XMLParser = XMLParser(fields=["think", "answer"]),
        funcs: List = [],
        weights: List[float] = [],
    ) -> None:

        super().__init__(funcs=funcs, weights=weights, parser=parser)

        self.parser = parser
        self.reward_funcs = [
            self.parser.get_format_reward_func(),
            self.accumulated_env_reward_func,
        ]
        self.reward_weights = [
            0.2,
            1.0,
        ]

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    def accumulated_env_reward_func(self, state: Dict[str, Any], **_) -> float:  # noqa: D401
        """Return the sum of rewards collected for the training player.

        The rollout logic in ``SotopiaEnv`` keeps track of the cumulative reward
        in ``state["reward_sum"]``.
        """
        try:
            return float(state.get("reward_sum", 0.0))
        except Exception:
            return 0.0 