from enum import Enum, StrEnum

from nfl.lib.datamodels import (
    Defensive_PenaltySpec,
    Offensive_PenaltySpec,
    PenaltySpec,
    Personal_FoulSpec,
)


class NFLTeam(StrEnum):
    PHI = "PHI"
    IND = "IND"
    NO = "NO"
    TEN = "TEN"
    WAS = "WAS"
    SEA = "SEA"
    CHI = "CHI"
    DEN = "DEN"
    ATL = "ATL"
    NYJ = "NYJ"
    TB = "TB"
    GB = "GB"
    CLE = "CLE"
    JAX = "JAX"
    MIA = "MIA"
    DAL = "DAL"
    KC = "KC"
    DET = "DET"
    NE = "NE"
    CAR = "CAR"
    SF  = "SF"
    BUF = "BUF"
    MIN = "MIN"
    BAL = "BAL"
    CIN = "CIN"
    NYG = "NYG"
    LA = "LA"
    LV = "LV"
    LAC = "LAC"
    PIT = "PIT"
    ARI = "ARI"
    HOU = "HOU"

class PlayType(StrEnum):
    PASS = "pass"
    RUN = "run"
    KICK = "kickoff"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"
    EXTRA_POINT = "extra_point"
    NO_PLAY = "no_play"
    QB_KNEEL = "qb_kneel"
    QB_SPIKE = "qb_spike"
    GAME_START = "game_start"

class PassType(StrEnum):
    SHORT = "short"
    DEEP = "deep"

class TeamType(StrEnum):
    HOME = "home"
    AWAY = "away"

# I made this up, we're going to use it for the NN
class PassOutcome(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INTERCEPTION = "interception"
    FUMBLE = "fumble"
    SACK = "sack"

class SeasonType(StrEnum):
    REGULAR = "REG"
    POSTSEASON = "POST"

class PenaltyType(Enum):
    FALSE_START = Offensive_PenaltySpec(
        name="False Start", penalty_distance=5, assessed_after_play=False
    )
    UNNECESSARY_ROUGHNESS = Personal_FoulSpec(
        name="Unnecessary Roughness",
        on_offense=True,
        on_defense=True,
        automatic_first_down=True,
    )
    ROUGHING_THE_PASSER = Personal_FoulSpec(
        name="Roughing the Passer",
        on_offense=False,
        on_defense=True,
        automatic_first_down=True,
    )
    OFFENSIVE_HOLDING = Offensive_PenaltySpec(
        name="Offensive Holding", penalty_distance=10, assessed_after_play=False
    )
    DEFENSIVE_OFFSIDE = Defensive_PenaltySpec(
        name="Defensive Offside", penalty_distance=5, assessed_after_play=False
    )
    ILLEGAL_FORMATION = PenaltySpec(
        name="Illegal Formation",
        penalty_distance=5,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
    )
    ILLEGAL_CONTACT = Defensive_PenaltySpec(
        name="Illegal Contact", penalty_distance=5, assessed_after_play=False, automatic_first_down=True
    )
    ENCROACHMENT = Defensive_PenaltySpec(
        name="Encroachment", penalty_distance=5, assessed_after_play=False
    )
    DELAY_OF_GAME = Offensive_PenaltySpec(
        name="Delay of Game",
        penalty_distance=5,
        assessed_after_play=False,
    )
    FAIR_CATCH_INTERFERENCE = Defensive_PenaltySpec(
        name="Fair Catch Interference", penalty_distance=15, assessed_after_play=False, enabled = False
    )
    DEFENSIVE_HOLDING = Defensive_PenaltySpec(
        name="Defensive Holding", penalty_distance=5, assessed_after_play=False, automatic_first_down=True
    )
    ILLEGAL_MOTION = Offensive_PenaltySpec(
        name="Illegal Motion", penalty_distance=5, assessed_after_play=False
    )
    UNSPORTSMANLIKE_CONDUCT = Personal_FoulSpec(
        name="Unsportsmanlike Conduct",
        on_offense=True,
        on_defense=True,
        automatic_first_down=True,
    )
    OFFENSIVE_PASS_INTERFERENCE = Offensive_PenaltySpec(
        name="Offensive Pass Interference",
        penalty_distance=10,
        assessed_after_play=False,
        loss_of_down=False,
    )
    ILLEGAL_BLOCK_ABOVE_THE_WAIST = PenaltySpec(
        name="Illegal Block Above the Waist",
        penalty_distance=10,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
    )
    DEFENSIVE_PASS_INTERFERENCE = Defensive_PenaltySpec(
        name="Defensive Pass Interference",
        penalty_distance=None,  # Spot foul
        assessed_after_play=False,
        automatic_first_down=True,
    )
    FACE_MASK = Personal_FoulSpec(
        name="Face Mask",
        on_offense=True,
        on_defense=True,
        automatic_first_down=True,
    )
    TRIPPING = PenaltySpec(
        name="Tripping",
        penalty_distance=15,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        automatic_first_down=True,  # Automatic 1st down if committed by defense
    )
    ILLEGAL_SHIFT = Offensive_PenaltySpec(
        name="Illegal Shift",
        penalty_distance=5,
        assessed_after_play=False,
    )
    INELIGIBLE_DOWNFIELD_PASS = Offensive_PenaltySpec(
        name="Ineligible Downfield Pass",
        penalty_distance=5,
        assessed_after_play=False,
    )
    ILLEGAL_SUBSTITUTION = PenaltySpec(
        name="Illegal Substitution",
        penalty_distance=5,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
    )
    INTENTIONAL_GROUNDING = Offensive_PenaltySpec(
        name="Intentional Grounding",
        penalty_distance=10,  # Or spot of foul if >10 yards behind LOS
        assessed_after_play=False,
        loss_of_down=True,
    )
    ILLEGAL_USE_OF_HANDS = PenaltySpec(
        name="Illegal Use of Hands",
        penalty_distance=10,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        automatic_first_down=True,  # If committed by defense
    )
    INELIGIBLE_DOWNFIELD_KICK = PenaltySpec(
        name="Ineligible Downfield Kick",
        penalty_distance=5,
        on_offense=True,  # Kicking team
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    KICKOFF_OUT_OF_BOUNDS = PenaltySpec(
        name="Kickoff Out of Bounds",
        penalty_distance=25,  # Placed at 40-yard line (25 yards from kickoff spot)
        on_offense=True,  # Kicking team
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    CLIPPING = PenaltySpec(
        name="Clipping",
        penalty_distance=15,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        automatic_first_down=True,  # If committed by defense
    )
    ILLEGAL_FORWARD_PASS = Offensive_PenaltySpec(
        name="Illegal Forward Pass",
        penalty_distance=5,
        assessed_after_play=False,
        loss_of_down=True,
    )
    ILLEGAL_TOUCH_KICK = PenaltySpec(
        name="Illegal Touch Kick",
        penalty_distance=0,  # Violation/spot of touch rather than yardage distance
        on_offense=True,  # Kicking team
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    NEUTRAL_ZONE_INFRACTION = Defensive_PenaltySpec(
        name="Neutral Zone Infraction",
        penalty_distance=5,
        assessed_after_play=False,  # Dead-ball foul before the snap
    )
    TAUNTING = Personal_FoulSpec(
        name="Taunting",
        on_offense=True,
        on_defense=True,
        automatic_first_down=True,
    )
    RUNNING_INTO_THE_KICKER = Defensive_PenaltySpec(
        name="Running Into the Kicker",
        penalty_distance=5,
        assessed_after_play=False,
        enabled=False
    )
    DISQUALIFICATION = Personal_FoulSpec(
        name="Disqualification",
        on_offense=True,
        on_defense=True,
        automatic_first_down=True,
    )
    OFFENSIVE_OFFSIDE = Offensive_PenaltySpec(
        name="Offensive Offside",
        penalty_distance=5,
        assessed_after_play=False,
    )
    ROUGHING_THE_KICKER = Defensive_PenaltySpec(
        name="Roughing the Kicker",
        penalty_distance=15,
        assessed_after_play=False,
        automatic_first_down=True,
        enabled=False
    )
    OFFSIDE_ON_FREE_KICK = PenaltySpec(
        name="Offside on Free Kick",
        penalty_distance=5,
        on_offense=True,  # Kicking team
        on_defense=True,  # Receiving team
        assessed_after_play=False,
        enabled=False
    )
    DEFENSIVE_DELAY_OF_GAME = Defensive_PenaltySpec(
        name="Defensive Delay of Game",
        penalty_distance=5,
        assessed_after_play=False,
    )
    CHOP_BLOCK = PenaltySpec(
        name="Chop Block",
        penalty_distance=15,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        automatic_first_down=True,  # If committed by defense
    )
    PERSONAL_FOUL = Personal_FoulSpec(
        name="Personal Foul",
        on_offense=True,
        on_defense=True,
        automatic_first_down=True,
    )
    ILLEGAL_CUT = PenaltySpec(
        name="Illegal Cut",
        penalty_distance=15,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        automatic_first_down=True,  # If committed by defense
    )
    ILLEGAL_CRACKBACK = PenaltySpec(
        name="Illegal Crackback",
        penalty_distance=15,
        on_offense=True,
        on_defense=False,
        assessed_after_play=False,
    )
    ILLEGAL_TOUCH_PASS = Offensive_PenaltySpec(
        name="Illegal Touch Pass",
        penalty_distance=5,
        assessed_after_play=False,
        loss_of_down=True,
    )
    ILLEGAL_BLINDSIDE_BLOCK = PenaltySpec(
        name="Illegal Blindside Block",
        penalty_distance=15,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        automatic_first_down=True,  # If committed by defense
    )
    SHORT_FREE_KICK = PenaltySpec(
        name="Short Free Kick",
        penalty_distance=5,
        on_offense=True,  # Kicking team
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    LEVERAGE = Defensive_PenaltySpec(
        name="Leverage",
        penalty_distance=15,
        assessed_after_play=False,
        automatic_first_down=True,
        enabled=False
    )
    ILLEGALLY_KICKING_BALL = PenaltySpec(
        name="Illegally Kicking Ball",
        penalty_distance=10,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        enabled=False
    )
    DELAY_OF_KICKOFF = PenaltySpec(
        name="Delay of Kickoff",
        penalty_distance=5,
        on_offense=True,  # Kicking team
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    INVALID_FAIR_CATCH_SIGNAL = PenaltySpec(
        name="Invalid Fair Catch Signal",
        penalty_distance=5,
        on_offense=False,
        on_defense=True,  # Receiving team making the signal
        assessed_after_play=False,
        enabled=False
    )
    LEAPING = Defensive_PenaltySpec(
        name="Leaping",
        penalty_distance=15,
        assessed_after_play=False,
        automatic_first_down=True,
        enabled=False
    )
    LOW_BLOCK = PenaltySpec(
        name="Low Block",
        penalty_distance=15,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        automatic_first_down=True,  # If committed by defense
    )
    INTERFERENCE_WITH_OPPORTUNITY_TO_CATCH = PenaltySpec(
        name="Interference with Opportunity to Catch",
        penalty_distance=15,
        on_offense=True,  # Kicking team
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    LOWERING_THE_HEAD_TO_MAKE_FORCIBLE_CONTACT = Personal_FoulSpec(
        name="Lowering the Head to Make Forcible Contact",
        on_offense=True,
        on_defense=True,
        automatic_first_down=True,
    )
    DEFENSIVE_TOO_MANY_MEN_ON_FIELD = Defensive_PenaltySpec(
        name="Defensive Too Many Men on Field",
        penalty_distance=5,
        assessed_after_play=False,
    )
    PLAYER_OUT_OF_BOUNDS_ON_KICK = PenaltySpec(
        name="Player Out of Bounds on Kick",
        penalty_distance=5,
        on_offense=True,  # Kicking team player stepping out voluntarily
        on_defense=False,
        assessed_after_play=False,
        enabled = False
    )
    HORSE_COLLAR_TACKLE = Personal_FoulSpec(
        name="Horse Collar Tackle",
        on_offense=True,
        on_defense=True,
        automatic_first_down=True,
    )
    ILLEGAL_PEELBACK = PenaltySpec(
        name="Illegal Peelback",
        penalty_distance=15,
        on_offense=True,
        on_defense=False,
        assessed_after_play=False,
    )
    OFFENSIVE_TOO_MANY_MEN_ON_FIELD = Offensive_PenaltySpec(
        name="Offensive Too Many Men on Field",
        penalty_distance=5,
        assessed_after_play=False,
    )
    KICK_CATCH_INTERFERENCE = PenaltySpec(
        name="Kick Catch Interference",
        penalty_distance=15,
        on_offense=True,  # Kicking team
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    ILLEGAL_DOUBLE_TEAM_BLOCK = PenaltySpec(
        name="Illegal Double-Team Block",
        penalty_distance=15,
        on_offense=True,  # Receiving team on kickoffs
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    ILLEGAL_KICK_KICKING_LOOSE_BALL = PenaltySpec(
        name="Illegal Kick/Kicking Loose Ball",
        penalty_distance=10,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
    )
    ILLEGAL_BAT = PenaltySpec(
        name="Illegal Bat",
        penalty_distance=10,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
    )
    LOWERING_THE_HEAD_TO_INITIATE_CONTACT = Personal_FoulSpec(
        name="Lowering the Head to Initiate Contact",
        on_offense=True,
        on_defense=True,
        automatic_first_down=True,
    )
    ILLEGAL_WEDGE = PenaltySpec(
        name="Illegal Wedge",
        penalty_distance=15,
        on_offense=True,  # Receiving team on kickoffs
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    KICKOFF_SHORT_OF_LANDING_ZONE = PenaltySpec(
        name="Kickoff Short of Landing Zone",
        penalty_distance=None,  # Treated as kickoff out of bounds
        on_offense=True,  # Kicking team
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    HIP_DROP_TACKLE = Personal_FoulSpec(
        name="Hip Drop Tackle",
        on_offense=False,
        on_defense=True,
        automatic_first_down=True,
    )
    ILLEGAL_SCRIMMAGE_KICK = PenaltySpec(
        name="Illegal Scrimmage Kick",
        penalty_distance=5,
        on_offense=True,
        on_defense=False,
        assessed_after_play=False,
        loss_of_down=True,
        enabled=False
    )
    ILLEGAL_KICK = PenaltySpec(
        name="Illegal Kick",
        penalty_distance=10,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        enabled=False
    )
    PLAYER_OUT_OF_BOUNDS_ON_PUNT = PenaltySpec(
        name="Player Out of Bounds on Punt",
        penalty_distance=5,
        on_offense=True,  # Kicking team gunner/player going out voluntarily
        on_defense=False,
        assessed_after_play=False,
        enabled=False
    )
    ILLGEAL_PROCEDURE = PenaltySpec(
        name="Illegal Procedure",
        penalty_distance=5,
        on_offense=True,
        on_defense=True,
        assessed_after_play=False,
        enabled=False
    )
    ILLEGAL_RECIEVER_PASS = Offensive_PenaltySpec(
        name="Illegal Receiver Pass",
        penalty_distance=5,
        assessed_after_play=False,
        loss_of_down=True,  # Ineligible player touching forward pass
    )

class SurfaceType(StrEnum):
    GRASS = "grass"
    FIELDTURF = "fieldturf"
    SPORTTURF = "sportturf"
    MATRIXTURF = "matrixurf"
    A_TURF = "a_turf"
    FIELDTURF_MISSPELLING = "fieldturf "
    ASTROPLAY = "astroplay"

class NFLPlayType(StrEnum):
    GAME_START = "GAME_START"
    KICK_OFF = "KICK_OFF"
    RUSH = "RUSH"
    PASS = "PASS"
    PUNT = "PUNT"
    PENALTY = "PENALTY"
    XP_KICK = "XP_KICK"
    TIMEOUT = "TIMEOUT"
    FIELD_GOAL = "FIELD_GOAL"
    SACK = "SACK"
    END_QUARTER = "END_QUARTER"
    INTERCEPTION = "INTERCEPTION"
    UNSPECIFIED = "UNSPECIFIED"
    END_GAME = "END_GAME"
    FUMBLE_RECOVERED_BY_OPPONENT = "FUMBLE_RECOVERED_BY_OPPONENT"
    PAT2 = "PAT2"
    COMMENT = "COMMENT"
    FREE_KICK = "FREE_KICK"

class RunLocations(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE ="middle"

class FieldGoalResults(StrEnum):
    MADE = "made"
    BLOCKED = "blocked"
    MISSED = "missed"

class RunGaps(StrEnum):
    END = "end"
    GUARD = "guard"
    TACKLE = "tackle"

class Result(StrEnum):
    FAILURE = "failure"
    SUCCESS = "success"

class Half(StrEnum):
    FIRST = "Half1"
    SECOND = "Half2"
    OVERTIME = "Overtime"

class DriveResult(StrEnum):
    PUNT = "Punt"
    TOUCHDOWN = "Touchdown"
    TURNOVER = "Turnover"
    FIELD_GOAL = "Field goal"
    MISSED_FIELD_GOAL = "Missed field goal"
    OPP_TOUCHDOWN = "Opp touchdown"
    END_OF_HALF = "End of half"
    SAFTEY = "Saftey"
    TURNOVER_ON_DOWNS = "Turnover on downs"

class SpecialTeamsPlayType(StrEnum):
    PENALTY = "Penalty"

class DriveStartReason(StrEnum):
    PUNT = "Punt"
    TOUCHDOWN = "Touchdown"
    INTERCEPTION = "Interception"
    FUMBLE = "Fumble"
    FIELD_GOAL = "Field Goal"
    BLOCKED_FG = "Blocked FG"
    END_OF_HALF = "End of Half"
    MISSED_FG = "Missed FG"
    SAFETY = "Safety"
    DOWNS = "Downs"
    FUMBLE_SAFETY = "Fumble, Safety"
    BLOCKED_PUNT = "Blocked Punt"
    BLOCKED_PUNT_DOWNS = "Blocked Punt, Downs"
    BLOCKED_FG_DOWNS = "Blocked FG, Downs"
    UNKNOWN = "UNKNOWN"
    KICKOFF = "KICKOFF"
    PUNT_UPPER = "PUNT"
    INTERCEPTION_UPPER = "INTERCEPTION"
    FUMBLE_UPPER = "FUMBLE"
    DOWNS_UPPER = "DOWNS"
    MISSED_FG_UPPER = "MISSED_FG"
    BLOCKED_FG_UPPER = "BLOCKED_FG"
    BLOCKED_FG_DOWNS_UPPER = "BLOCKED_FG,_DOWNS"
    MUFFED_PUNT = "MUFFED_PUNT"
    ONSIDE_KICK = "ONSIDE_KICK"
    BLOCKED_PUNT_UPPER = "BLOCKED_PUNT"
    BLOCKED_PUNT_DOWNS_UPPER = "BLOCKED_PUNT,_DOWNS"
    MUFFED_KICKOFF = "MUFFED_KICKOFF"
    BLOCKED_PUNT_DOWNS_ALT = "BLOCKED_PUNT_DOWNS"
    OWN_KICKOFF = "OWN_KICKOFF"
    MUFFED_FG = "MUFFED_FG"
    BLOCKED_FG_DOWNS_ALT = "BLOCKED_FG_DOWNS"

class RoofType(StrEnum):
    OUTDOORS = "outdoors"
    CLOSED = "closed"
    DOME = "dome"
    OPEN = "open"
