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
        if random.random() >= 0.10:
            return []

        penalty_type = random.choice(list(PenaltyType))
        spec = penalty_type.value

        is_on_offense = self._sample_penalty_side(spec)
        distance = spec.penalty_distance if spec.penalty_distance is not None else 15

        return [PenaltyHelper(penalty=penalty_type, is_on_offense=is_on_offense, distance=distance)]

    def get_play_outcome(
        self, scenario: Scenario, play: torch.Tensor
    ) -> TransitionResult:
        penalties = self._generate_penalty(scenario)

        # see if there's a turnover too
        # if turnover + off penalty -> turnover
        # if turnover + def penalty -> def penalty
        # if off penalty and 4th down -> turnover
        # if off penalty, don't bother to run the play. Return new scenario.
        # Run play to get outcome, assuming no penalties
        # Depends on the penalty otherwise. If off yds > def penalty -> off yds and ignore penalty
        # if def penalty and after play, add them up
        # at this point should be no penalties
        # run play, can yeild yards, lose yards, incomplete, etc. Will want to run clock off here too. 
        # see if play went far enough to get touchdown, saftey, etc. will need to handle those.
        # after running down the clock will need to see if we are at end of half or game and handle those.
        # return resulting TransitionResult