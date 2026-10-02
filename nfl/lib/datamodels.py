from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class PenaltySpec:
    name: str = ""
    penalty_distance: int | None = None
    on_offense: bool = False
    on_defense: bool = False
    assessed_after_play: bool = False
    loss_of_down: bool = False
    automatic_first_down: bool = False
    enabled: bool = True

@dataclass(frozen=True)
class Defensive_PenaltySpec(PenaltySpec):
    on_offense: bool = False
    on_defense: bool = True

@dataclass(frozen=True)
class Offensive_PenaltySpec(PenaltySpec):
    on_offense: bool = True
    on_defense: bool = False

@dataclass(frozen=True)
class Personal_FoulSpec(PenaltySpec):
    penalty_distance: int = 15
    assessed_after_play: bool = True

@dataclass
class Scenario:
    yardline: float
    seconds_remaining: float
    has_turf: bool | int
    temp: float
    wind: float
    hs_roof: bool | int
    yds_to_go: bool | int
    score_to_go: float
    goal_differential: float
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
    is_touchdown: bool = False
    is_turnover: bool = False
    is_safety: bool = False
    is_game_over: bool = False
    penalty: PenaltySpec | None = None
    