from dataclasses import dataclass
from nfl.lib.nfl_enums import PenaltyType
import torch

@dataclass
class Penalty:
    penalty_type: PenaltyType
    penalty_yards: int
    penalty_on_offence: bool


@dataclass
class Scenario:
    yardline: float
    seconds_remaining: float
    has_turf: bool | int
    temp: float
    wind: float
    has_roof: bool | int
    yds_to_go: float
    goal_to_go: bool | int
    score_differential: float
    down: int
    div_game: bool | int
    day_of_season: int
    series: int

    def to_tensor(self):
        return torch.tensor([self.yardline, self.seconds_remaining, self.has_turf, self.temp, self.wind, self.has_roof, self.yds_to_go, self.goal_to_go, self.score_differential, self.down, self.div_game, self.day_of_season, self.series], dtype=torch.float32)


@dataclass
class TransitionResult:
    # State transition output after evaluating play outcome
    next_scenario: Scenario
    is_touchdown: bool
    is_turnover: bool
    is_safety: bool
    is_game_over: bool
    penalty: Penalty
    