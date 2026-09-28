from dataclasses import dataclass, replace
from typing import Tuple
import random
import torch

from nfl.lib.datamodels import Scenario, TransitionResult
from nfl.lib.nfl_enums import PenaltyType


@dataclass
class PenaltyHelper:
    penalty: PenaltyType
    is_on_offense: bool
    distance: int


class PlaySimulator:
    def __init__(self):
        pass

    def _sample_penalty_side(self, spec) -> bool:
        """Determines if the penalty is committed by the offense."""
        if spec.on_offense and spec.on_defense:
            return random.choice([True, False])
        return spec.on_offense

    def _generate_penalty(self, scenario: Scenario) -> list[PenaltyHelper]:
        # 10% penalty occurrence rate
        if random.random() >= 0.10:
            return []

        penalty_type = random.choice(list(PenaltyType))
        spec = penalty_type.value

        is_on_offense = self._sample_penalty_side(spec)
        distance = spec.penalty_distance if spec.penalty_distance is not None else 15

        return [PenaltyHelper(penalty=penalty_type, is_on_offense=is_on_offense, distance=distance)]

    def get_play_outcome(
        self, scenario: Scenario, play: torch.Tensor
    ) -> Tuple[Scenario, TransitionResult]:
        penalties = self._generate_penalty(scenario)
        if not penalties:
            # Handle normal play resolution...
            return scenario, TransitionResult()

        off_penalties = [p for p in penalties if p.is_on_offense]
        def_penalties = [p for p in penalties if not p.is_on_offense]

        # Offsetting penalties cancel out
        if off_penalties and def_penalties:
            return scenario, TransitionResult()

        if off_penalties:
            assessed = max(off_penalties, key=lambda p: p.distance)
            spec = assessed.penalty.value

            is_loss_of_down = spec.loss_of_down
            turnover_on_downs = scenario.down == 4 and is_loss_of_down

            if turnover_on_downs:
                next_down = 1
                next_yds_to_go = 10
            else:
                next_down = scenario.down if is_loss_of_down else scenario.down + 1
                next_yds_to_go = scenario.yds_to_go + assessed.distance

            scenario = replace(
                scenario,
                yardline=scenario.yardline - assessed.distance,
                yds_to_go=next_yds_to_go,
                down=next_down,
            )

        return scenario, TransitionResult()