import torch
from typing import Tuple
from nfl.lib.datamodels import TransitionResult, Scenario, Penalty
import random
from nfl.lib.nfl_enums import PenaltyType

class PlaySimulator:
    def __init__(self):
        pass

    def _generate_penalty(self, scenario: Scenario) -> list[Penalty] | None:
        # given a scenario, generate penalties. This just randomly picks and is not good.
        if random.random() < 0.1:
            penalty_type = random.choice(list(PenaltyType))
            penalty_distance = random.choice([5,10,15])
            return [Penalty(penalty_type=penalty_type,
                            penalty_yards=penalty_distance,
                            penalty_on_offense=False,
                            assessed_after_play=False)]
        return None
        

    def get_play_outcome(self, scenario: Scenario, play: torch.Tensor) -> Tuple[Scenario, TransitionResult]:
        # first, generate new things independent of the playcall. 
        # Penalty???

        # Then get result of the play
        # should include if 