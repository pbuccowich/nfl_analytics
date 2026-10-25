from dataclasses import asdict, dataclass


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
    """Game state before a play; field names match SCENARIO_COLS in the data."""

    yardline_100: float
    game_seconds_remaining: float
    down: int
    ydstogo: float
    goal_to_go: bool
    score_differential: float
    series: int = 1
    div_game: bool = False
    temp: float = 60.0
    wind: float = 5.0
    day_of_season: int = 60
    has_roof: bool = False
    has_turf: bool = False
    # The models never trained on TEST_SEASON (2025), so it's the season to play in.
    # A team of None encodes as no franchise at all (all-zero one-hot), which the
    # models never saw in training; pass real teams for meaningful predictions.
    season: int = 2025
    posteam: str | None = None
    defteam: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            k: v if v is None or isinstance(v, str) else float(v)
            for k, v in asdict(self).items()
        }


@dataclass
class TransitionResult:
    # State transition output after evaluating play outcome
    next_scenario: Scenario
    is_touchdown: bool = False
    is_turnover: bool = False
    is_safety: bool = False
    is_game_over: bool = False
    penalty: PenaltySpec | None = None
