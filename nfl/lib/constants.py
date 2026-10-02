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
    "series"
]

PLAY_COLS = [
    "play_type",
    "pass_location",
#    "pass_length",
#    "run_location",
    "run_gap",
    "shotgun",
    "no_huddle",
    "qb_kneel",
    "qb_spike",
    "qb_scramble",
    "air_yards"
]

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

EXCLUDED_PLAY_TYPES = {
    enums.PlayType.KICK,
    enums.PlayType.EXTRA_POINT,
    enums.PlayType.NO_PLAY,
    enums.PlayType.GAME_START,
}

POSSIBLE_PENALTIES = [
    penalty for penalty in enums.PenaltyType if penalty.value.enabled
]

PENALTY_ENUM_MAPPING = {member.name: member for member in enums.PenaltyType}
PENALTY_ENUM_MAPPING = {
    "Defensive 12 On-field": enums.PenaltyType.DEFENSIVE_TOO_MANY_MEN_ON_FIELD,
    "Offensive 12 On-field": enums.PenaltyType.OFFENSIVE_TOO_MANY_MEN_ON_FIELD,
    "Horse Collar": enums.PenaltyType.HORSE_COLLAR_TACKLE,
    "Face Mask (5 Yards)": enums.PenaltyType.FACE_MASK
}