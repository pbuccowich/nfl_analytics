from nfl.lib import nfl_enums as enums

SCENARIO_COLS = [
    "yardline_100",
    "game_seconds_remaining",
    "has_turf",
    "temp",
    "wind",
    "has_roof",
    "ydstogo",
    "goal_to_go",
    "score_differential",
    "down",
    "div_game",
    "day_of_season",
    "series",
]

PLAY_COLS = [
    "play_type",
    "pass_location",
    "pass_length",
    "run_location",
    "run_gap",
    "shotgun",
    "no_huddle",
    "qb_dropback",
    "qb_kneel",
    "qb_spike",
]

# Categorical play columns, one-hot encoded against these fixed categories so the
# feature columns never depend on which values happen to be in the data
CATEGORICAL_PLAY_COLS = {
    "play_type": [p.value for p in enums.PlayType],
    # Pass and run directions share the same left/middle/right values
    "pass_location": [d.value for d in enums.RunLocations],
    "run_location": [d.value for d in enums.RunLocations],
}

RESULT_COLS = [
    "epa",
    "wpa",
    "success",
    "result",
    "series_success",
    "tackle_for_loss",
    "saftey",
    "yards_gained",
    "touchdown",
    "fumble",
    "complete_pass",
    "rushing_yards",
    "fumble_lost",
    "interception",
    "sack",
    "penalty_yards",
]

# Yards gained on passes and runs are predicted as a distribution over these
# whole-yard bins (inclusive). The data's extremes are a 30-yard sack and a 99-yard
# catch; rarer bigger losses are censored as "at most -30".
YARDS_RANGE = (-30, 99)

# Pass outcomes that end with the ball carrier down somewhere, so they need yards
PASS_YARDS_OUTCOMES = [
    enums.PassOutcome.COMPLETE,
    enums.PassOutcome.SCRAMBLE,
    enums.PassOutcome.SACK,
]

# (penalty, committed by offense) classes the penalty models predict, each with at
# least 300 cases in 1999-2025. The models also predict "no penalty" and an "other"
# class per side for everything rarer. Pre-snap fouls happen before any call is
# known; live-ball fouls happen on a snapped pass or run.
PRE_SNAP_PENALTIES = [
    (enums.PenaltyType.FALSE_START, True),
    (enums.PenaltyType.DELAY_OF_GAME, True),
    (enums.PenaltyType.NEUTRAL_ZONE_INFRACTION, False),
    (enums.PenaltyType.ENCROACHMENT, False),
    (enums.PenaltyType.DEFENSIVE_OFFSIDE, False),
]
LIVE_BALL_PENALTIES = [
    (enums.PenaltyType.OFFENSIVE_HOLDING, True),
    (enums.PenaltyType.DEFENSIVE_PASS_INTERFERENCE, False),
    (enums.PenaltyType.DEFENSIVE_HOLDING, False),
    (enums.PenaltyType.DEFENSIVE_OFFSIDE, False),
    (enums.PenaltyType.UNNECESSARY_ROUGHNESS, False),
    (enums.PenaltyType.ROUGHING_THE_PASSER, False),
    (enums.PenaltyType.OFFENSIVE_PASS_INTERFERENCE, True),
    (enums.PenaltyType.ILLEGAL_CONTACT, False),
    (enums.PenaltyType.FACE_MASK, False),
    (enums.PenaltyType.ILLEGAL_USE_OF_HANDS, False),
    (enums.PenaltyType.INTENTIONAL_GROUNDING, True),
    (enums.PenaltyType.ILLEGAL_FORMATION, True),
    (enums.PenaltyType.UNNECESSARY_ROUGHNESS, True),
    (enums.PenaltyType.ILLEGAL_SHIFT, True),
    (enums.PenaltyType.INELIGIBLE_DOWNFIELD_PASS, True),
    (enums.PenaltyType.ILLEGAL_BLOCK_ABOVE_THE_WAIST, True),
    (enums.PenaltyType.ILLEGAL_USE_OF_HANDS, True),
    (enums.PenaltyType.UNSPORTSMANLIKE_CONDUCT, False),
    (enums.PenaltyType.PERSONAL_FOUL, False),
    (enums.PenaltyType.UNSPORTSMANLIKE_CONDUCT, True),
    (enums.PenaltyType.FACE_MASK, True),
]

EXCLUDED_PLAY_TYPES = {
    enums.PlayType.KICK,
    enums.PlayType.EXTRA_POINT,
    enums.PlayType.NO_PLAY,
    enums.PlayType.GAME_START,
}

POSSIBLE_PENALTIES = [penalty for penalty in enums.PenaltyType if penalty.value.enabled]

PENALTY_ENUM_MAPPING = {member.value.name: member for member in enums.PenaltyType}
PENALTY_ENUM_MAPPING.update(
    {
        "Defensive 12 On-field": enums.PenaltyType.DEFENSIVE_TOO_MANY_MEN_ON_FIELD,
        "Offensive 12 On-field": enums.PenaltyType.OFFENSIVE_TOO_MANY_MEN_ON_FIELD,
        "Horse Collar": enums.PenaltyType.HORSE_COLLAR_TACKLE,
        "Face Mask (5 Yards)": enums.PenaltyType.FACE_MASK,
    }
)
