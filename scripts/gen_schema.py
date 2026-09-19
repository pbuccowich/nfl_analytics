import ast
import json
from nfl.lib import enums

raw_str = "{'play_id': 'Column(int64)', 'game_id': 'Column(object)', 'old_game_id': 'Column(int64)', 'home_team': 'Column(object)', 'away_team': 'Column(object)', 'season_type': 'Column(object)', 'week': 'Column(int64)', 'posteam': 'Column(object)', 'posteam_type': 'Column(object)', 'defteam': 'Column(object)', 'side_of_field': 'Column(object)', 'yardline_100': 'Column(float64)', 'game_date': 'Column(datetime64[ns])', 'quarter_seconds_remaining': 'Column(float64)', 'half_seconds_remaining': 'Column(float64)', 'game_seconds_remaining': 'Column(float64)', 'game_half': 'Column(object)', 'quarter_end': 'Column(bool)', 'drive': 'Column(float64)', 'sp': 'Column(int64)', 'qtr': 'Column(int64)', 'down': 'Column(float64)', 'goal_to_go': 'Column(float64)', 'time': 'Column(object)', 'yrdln': 'Column(object)', 'ydstogo': 'Column(int64)', 'ydsnet': 'Column(float64)', 'desc': 'Column(object)', 'play_type': 'Column(object)', 'yards_gained': 'Column(float64)', 'shotgun': 'Column(int64)', 'no_huddle': 'Column(int64)', 'qb_dropback': 'Column(float64)', 'qb_kneel': 'Column(int64)', 'qb_spike': 'Column(int64)', 'qb_scramble': 'Column(int64)', 'pass_length': 'Column(object)', 'pass_location': 'Column(object)', 'air_yards': 'Column(float64)', 'yards_after_catch': 'Column(float64)', 'run_location': 'Column(object)', 'run_gap': 'Column(object)', 'field_goal_result': 'Column(object)', 'kick_distance': 'Column(float64)', 'extra_point_result': 'Column(object)', 'two_point_conv_result': 'Column(object)', 'home_timeouts_remaining': 'Column(int64)', 'away_timeouts_remaining': 'Column(int64)', 'timeout': 'Column(float64)', 'timeout_team': 'Column(object)', 'td_team': 'Column(object)', 'td_player_name': 'Column(object)', 'td_player_id': 'Column(object)', 'posteam_timeouts_remaining': 'Column(float64)', 'defteam_timeouts_remaining': 'Column(float64)', 'total_home_score': 'Column(int64)', 'total_away_score': 'Column(int64)', 'posteam_score': 'Column(float64)', 'defteam_score': 'Column(float64)', 'score_differential': 'Column(float64)', 'posteam_score_post': 'Column(float64)', 'defteam_score_post': 'Column(float64)', 'score_differential_post': 'Column(float64)', 'no_score_prob': 'Column(float64)', 'opp_fg_prob': 'Column(float64)', 'opp_safety_prob': 'Column(float64)', 'opp_td_prob': 'Column(float64)', 'fg_prob': 'Column(float64)', 'safety_prob': 'Column(float64)', 'td_prob': 'Column(float64)', 'extra_point_prob': 'Column(float64)', 'two_point_conversion_prob': 'Column(float64)', 'ep': 'Column(float64)', 'epa': 'Column(float64)', 'total_home_epa': 'Column(float64)', 'total_away_epa': 'Column(float64)', 'total_home_rush_epa': 'Column(float64)', 'total_away_rush_epa': 'Column(float64)', 'total_home_pass_epa': 'Column(float64)', 'total_away_pass_epa': 'Column(float64)', 'air_epa': 'Column(float64)', 'yac_epa': 'Column(float64)', 'comp_air_epa': 'Column(float64)', 'comp_yac_epa': 'Column(float64)', 'total_home_comp_air_epa': 'Column(float64)', 'total_away_comp_air_epa': 'Column(float64)', 'total_home_comp_yac_epa': 'Column(float64)', 'total_away_comp_yac_epa': 'Column(float64)', 'total_home_raw_air_epa': 'Column(float64)', 'total_away_raw_air_epa': 'Column(float64)', 'total_home_raw_yac_epa': 'Column(float64)', 'total_away_raw_yac_epa': 'Column(float64)', 'wp': 'Column(float64)', 'def_wp': 'Column(float64)', 'home_wp': 'Column(float64)', 'away_wp': 'Column(float64)', 'wpa': 'Column(float64)', 'vegas_wpa': 'Column(float64)', 'vegas_home_wpa': 'Column(float64)', 'home_wp_post': 'Column(float64)', 'away_wp_post': 'Column(float64)', 'vegas_wp': 'Column(float64)', 'vegas_home_wp': 'Column(float64)', 'total_home_rush_wpa': 'Column(float64)', 'total_away_rush_wpa': 'Column(float64)', 'total_home_pass_wpa': 'Column(float64)', 'total_away_pass_wpa': 'Column(float64)', 'air_wpa': 'Column(float64)', 'yac_wpa': 'Column(float64)', 'comp_air_wpa': 'Column(float64)', 'comp_yac_wpa': 'Column(float64)', 'total_home_comp_air_wpa': 'Column(float64)', 'total_away_comp_air_wpa': 'Column(float64)', 'total_home_comp_yac_wpa': 'Column(float64)', 'total_away_comp_yac_wpa': 'Column(float64)', 'total_home_raw_air_wpa': 'Column(float64)', 'total_away_raw_air_wpa': 'Column(float64)', 'total_home_raw_yac_wpa': 'Column(float64)', 'total_away_raw_yac_wpa': 'Column(float64)', 'punt_blocked': 'Column(float64)', 'first_down_rush': 'Column(float64)', 'first_down_pass': 'Column(float64)', 'first_down_penalty': 'Column(float64)', 'third_down_converted': 'Column(float64)', 'third_down_failed': 'Column(float64)', 'fourth_down_converted': 'Column(float64)', 'fourth_down_failed': 'Column(float64)', 'incomplete_pass': 'Column(float64)', 'touchback': 'Column(int64)', 'interception': 'Column(float64)', 'punt_inside_twenty': 'Column(float64)', 'punt_in_endzone': 'Column(float64)', 'punt_out_of_bounds': 'Column(float64)', 'punt_downed': 'Column(float64)', 'punt_fair_catch': 'Column(float64)', 'kickoff_inside_twenty': 'Column(float64)', 'kickoff_in_endzone': 'Column(float64)', 'kickoff_out_of_bounds': 'Column(float64)', 'kickoff_downed': 'Column(float64)', 'kickoff_fair_catch': 'Column(float64)', 'fumble_forced': 'Column(float64)', 'fumble_not_forced': 'Column(float64)', 'fumble_out_of_bounds': 'Column(float64)', 'solo_tackle': 'Column(float64)', 'safety': 'Column(float64)', 'penalty': 'Column(float64)', 'tackled_for_loss': 'Column(float64)', 'fumble_lost': 'Column(float64)', 'own_kickoff_recovery': 'Column(float64)', 'own_kickoff_recovery_td': 'Column(float64)', 'qb_hit': 'Column(float64)', 'rush_attempt': 'Column(float64)', 'pass_attempt': 'Column(float64)', 'sack': 'Column(float64)', 'touchdown': 'Column(float64)', 'pass_touchdown': 'Column(float64)', 'rush_touchdown': 'Column(float64)', 'return_touchdown': 'Column(float64)', 'extra_point_attempt': 'Column(float64)', 'two_point_attempt': 'Column(float64)', 'field_goal_attempt': 'Column(float64)', 'kickoff_attempt': 'Column(float64)', 'punt_attempt': 'Column(float64)', 'fumble': 'Column(float64)', 'complete_pass': 'Column(float64)', 'assist_tackle': 'Column(float64)', 'lateral_reception': 'Column(float64)', 'lateral_rush': 'Column(float64)', 'lateral_return': 'Column(float64)', 'lateral_recovery': 'Column(float64)', 'passer_player_id': 'Column(object)', 'passer_player_name': 'Column(object)', 'passing_yards': 'Column(float64)', 'receiver_player_id': 'Column(object)', 'receiver_player_name': 'Column(object)', 'receiving_yards': 'Column(float64)', 'rusher_player_id': 'Column(object)', 'rusher_player_name': 'Column(object)', 'rushing_yards': 'Column(float64)', 'lateral_receiver_player_id': 'Column(object)', 'lateral_receiver_player_name': 'Column(object)', 'lateral_receiving_yards': 'Column(float64)', 'lateral_rusher_player_id': 'Column(object)', 'lateral_rusher_player_name': 'Column(object)', 'lateral_rushing_yards': 'Column(float64)', 'lateral_sack_player_id': 'Column(float64)', 'lateral_sack_player_name': 'Column(float64)', 'interception_player_id': 'Column(object)', 'interception_player_name': 'Column(object)', 'lateral_interception_player_id': 'Column(object)', 'lateral_interception_player_name': 'Column(object)', 'punt_returner_player_id': 'Column(object)', 'punt_returner_player_name': 'Column(object)', 'lateral_punt_returner_player_id': 'Column(object)', 'lateral_punt_returner_player_name': 'Column(object)', 'kickoff_returner_player_name': 'Column(object)', 'kickoff_returner_player_id': 'Column(object)', 'lateral_kickoff_returner_player_id': 'Column(object)', 'lateral_kickoff_returner_player_name': 'Column(object)', 'punter_player_id': 'Column(object)', 'punter_player_name': 'Column(object)', 'kicker_player_name': 'Column(object)', 'kicker_player_id': 'Column(object)', 'own_kickoff_recovery_player_id': 'Column(object)', 'own_kickoff_recovery_player_name': 'Column(object)', 'blocked_player_id': 'Column(object)', 'blocked_player_name': 'Column(object)', 'tackle_for_loss_1_player_id': 'Column(object)', 'tackle_for_loss_1_player_name': 'Column(object)', 'tackle_for_loss_2_player_id': 'Column(object)', 'tackle_for_loss_2_player_name': 'Column(object)', 'qb_hit_1_player_id': 'Column(object)', 'qb_hit_1_player_name': 'Column(object)', 'qb_hit_2_player_id': 'Column(object)', 'qb_hit_2_player_name': 'Column(object)', 'forced_fumble_player_1_team': 'Column(object)', 'forced_fumble_player_1_player_id': 'Column(object)', 'forced_fumble_player_1_player_name': 'Column(object)', 'forced_fumble_player_2_team': 'Column(object)', 'forced_fumble_player_2_player_id': 'Column(object)', 'forced_fumble_player_2_player_name': 'Column(object)', 'solo_tackle_1_team': 'Column(object)', 'solo_tackle_2_team': 'Column(object)', 'solo_tackle_1_player_id': 'Column(object)', 'solo_tackle_2_player_id': 'Column(object)', 'solo_tackle_1_player_name': 'Column(object)', 'solo_tackle_2_player_name': 'Column(object)', 'assist_tackle_1_player_id': 'Column(object)', 'assist_tackle_1_player_name': 'Column(object)', 'assist_tackle_1_team': 'Column(object)', 'assist_tackle_2_player_id': 'Column(object)', 'assist_tackle_2_player_name': 'Column(object)', 'assist_tackle_2_team': 'Column(object)', 'assist_tackle_3_player_id': 'Column(object)', 'assist_tackle_3_player_name': 'Column(object)', 'assist_tackle_3_team': 'Column(object)', 'assist_tackle_4_player_id': 'Column(object)', 'assist_tackle_4_player_name': 'Column(object)', 'assist_tackle_4_team': 'Column(object)', 'tackle_with_assist': 'Column(float64)', 'tackle_with_assist_1_player_id': 'Column(object)', 'tackle_with_assist_1_player_name': 'Column(object)', 'tackle_with_assist_1_team': 'Column(object)', 'tackle_with_assist_2_player_id': 'Column(object)', 'tackle_with_assist_2_player_name': 'Column(object)', 'tackle_with_assist_2_team': 'Column(object)', 'pass_defense_1_player_id': 'Column(object)', 'pass_defense_1_player_name': 'Column(object)', 'pass_defense_2_player_id': 'Column(object)', 'pass_defense_2_player_name': 'Column(object)', 'fumbled_1_team': 'Column(object)', 'fumbled_1_player_id': 'Column(object)', 'fumbled_1_player_name': 'Column(object)', 'fumbled_2_player_id': 'Column(object)', 'fumbled_2_player_name': 'Column(object)', 'fumbled_2_team': 'Column(object)', 'fumble_recovery_1_team': 'Column(object)', 'fumble_recovery_1_yards': 'Column(float64)', 'fumble_recovery_1_player_id': 'Column(object)', 'fumble_recovery_1_player_name': 'Column(object)', 'fumble_recovery_2_team': 'Column(object)', 'fumble_recovery_2_yards': 'Column(float64)', 'fumble_recovery_2_player_id': 'Column(object)', 'fumble_recovery_2_player_name': 'Column(object)', 'sack_player_id': 'Column(object)', 'sack_player_name': 'Column(object)', 'half_sack_1_player_id': 'Column(object)', 'half_sack_1_player_name': 'Column(object)', 'half_sack_2_player_id': 'Column(object)', 'half_sack_2_player_name': 'Column(object)', 'return_team': 'Column(object)', 'return_yards': 'Column(float64)', 'penalty_team': 'Column(object)', 'penalty_player_id': 'Column(object)', 'penalty_player_name': 'Column(object)', 'penalty_yards': 'Column(float64)', 'replay_or_challenge': 'Column(int64)', 'replay_or_challenge_result': 'Column(object)', 'penalty_type': 'Column(object)', 'defensive_two_point_attempt': 'Column(float64)', 'defensive_two_point_conv': 'Column(float64)', 'defensive_extra_point_attempt': 'Column(float64)', 'defensive_extra_point_conv': 'Column(float64)', 'safety_player_name': 'Column(object)', 'safety_player_id': 'Column(object)', 'season': 'Column(int64)', 'cp': 'Column(float64)', 'cpoe': 'Column(float64)', 'series': 'Column(int64)', 'series_success': 'Column(int64)', 'series_result': 'Column(object)', 'order_sequence': 'Column(float64)', 'start_time': 'Column(object)', 'time_of_day': 'Column(object)', 'stadium': 'Column(object)', 'weather': 'Column(object)', 'nfl_api_id': 'Column(object)', 'play_clock': 'Column(float64)', 'play_deleted': 'Column(float64)', 'play_type_nfl': 'Column(object)', 'special_teams_play': 'Column(float64)', 'st_play_type': 'Column(object)', 'end_clock_time': 'Column(object)', 'end_yard_line': 'Column(object)', 'fixed_drive': 'Column(int64)', 'fixed_drive_result': 'Column(object)', 'drive_real_start_time': 'Column(object)', 'drive_play_count': 'Column(float64)', 'drive_time_of_possession': 'Column(object)', 'drive_first_downs': 'Column(float64)', 'drive_inside20': 'Column(float64)', 'drive_ended_with_score': 'Column(float64)', 'drive_quarter_start': 'Column(float64)', 'drive_quarter_end': 'Column(float64)', 'drive_yards_penalized': 'Column(float64)', 'drive_start_transition': 'Column(object)', 'drive_end_transition': 'Column(object)', 'drive_game_clock_start': 'Column(object)', 'drive_game_clock_end': 'Column(object)', 'drive_start_yard_line': 'Column(object)', 'drive_end_yard_line': 'Column(object)', 'drive_play_id_started': 'Column(float64)', 'drive_play_id_ended': 'Column(float64)', 'away_score': 'Column(int64)', 'home_score': 'Column(int64)', 'location': 'Column(object)', 'result': 'Column(int64)', 'total': 'Column(int64)', 'spread_line': 'Column(float64)', 'total_line': 'Column(float64)', 'div_game': 'Column(int64)', 'roof': 'Column(object)', 'surface': 'Column(object)', 'temp': 'Column(float64)', 'wind': 'Column(float64)', 'home_coach': 'Column(object)', 'away_coach': 'Column(object)', 'stadium_id': 'Column(object)', 'game_stadium': 'Column(object)', 'aborted_play': 'Column(bool)', 'success': 'Column(float64)', 'passer': 'Column(object)', 'passer_jersey_number': 'Column(float64)', 'rusher': 'Column(object)', 'rusher_jersey_number': 'Column(float64)', 'receiver': 'Column(object)', 'receiver_jersey_number': 'Column(float64)', 'pass': 'Column(int64)', 'rush': 'Column(int64)', 'first_down': 'Column(float64)', 'special': 'Column(int64)', 'play': 'Column(int64)', 'passer_id': 'Column(object)', 'rusher_id': 'Column(object)', 'receiver_id': 'Column(object)', 'name': 'Column(object)', 'jersey_number': 'Column(float64)', 'id': 'Column(object)', 'fantasy_player_name': 'Column(object)', 'fantasy_player_id': 'Column(object)', 'fantasy': 'Column(object)', 'fantasy_id': 'Column(object)', 'out_of_bounds': 'Column(int64)', 'home_opening_kickoff': 'Column(int64)', 'qb_epa': 'Column(float64)', 'xyac_epa': 'Column(float64)', 'xyac_mean_yardage': 'Column(float64)', 'xyac_median_yardage': 'Column(float64)', 'xyac_success': 'Column(float64)', 'xyac_fd': 'Column(float64)', 'xpass': 'Column(float64)', 'pass_oe': 'Column(float64)', 'year': 'Column(object)'}"
EXCLUDE_PHRASES = ["jersey_number", "fantasy", "player_id", "player_name", "old_game_id"]

NON_NULLABLE = []

TYPE_LOOKUP = {'play_id': 'int64',
               'yardline_100': 'int64',
               'air_yards': 'float64',
               'yards_after_catch': 'float64',
               'home_team': 'str', 
               'away_team':'str', 
               'season_type': 'str',
               'posteam': 'str',
               'posteam_type': 'str',
               'defteam': 'str',
               'location': 'str',
               'side_of_field': 'str',
               'qb_kneel': 'bool',
               'qb_spike': 'bool',
               'aborted_play': 'bool',
               'qb_dropback': 'bool',
               'no_huddle': 'bool',
               'shotgun': 'bool',
               'timeout': 'bool',
               'quarter_end': 'bool',
               'sp': 'bool',
               'goal_to_go': 'bool',
               'special': 'bool',
               'play': 'bool',
               'out_of_bounds': 'bool',
               'home_opening_kickoff': 'bool',
               'success': 'bool',
               'aborted_play': 'bool',
               'div_game': 'bool',
               'roof': 'str',
               'drive_inside_20': 'bool',
               'fixed_drive_result': 'str',
               'play_deleted': 'bool',
               }

ENUM_MAP = {
    "home_team": enums.NFLTeam,
    "away_team": enums.NFLTeam,
    "season_type": enums.SeasonType,
    "posteam": enums.NFLTeam,
    "posteam_type": enums.TeamType,
    "defteam": enums.NFLTeam,
    "side_of_field": enums.NFLTeam,
    "game_half": enums.Half,
    "fixed_drive_result": enums.DriveResult,
    "special_teams_play_type": enums.SpecialTeamsPlayType,
    "drive_start_transition": enums.DriveStartReason,
    "play_type": enums.PlayType,
    "pass_length": enums.PassType,
    "team_type": enums.TeamType,
    "penalty_type": enums.PenaltyType,
    "surface_type": enums.SurfaceType,
    "nfl_play_type": enums.NFLPlayType,
    "run_location": enums.RunLocations,
    "run_gap": enums.RunGaps,
    "half": enums.Half,
    "drive_result": enums.DriveResult,
    "drive_start_reason": enums.DriveStartReason,

}

def get_column_config(col_name, raw_dtype):
    # Determine type
    dtype = TYPE_LOOKUP.get(col_name, raw_dtype.replace('Column(', '').replace(')', ''))
    
    config = {
        "type": dtype or "object",
        "nullable": col_name not in NON_NULLABLE
    }
    
    # Assign Enums
    for pattern, enum_cls in ENUM_MAP.items():
        if pattern in col_name:
            config["ref"] = pattern
            break
            
    return config

# 3. Main execution
raw_data = ast.literal_eval(raw_str)
cleaned_dict = {
    k: get_column_config(k, v) 
    for k, v in raw_data.items() 
    if not any(phrase in k for phrase in ["jersey", "fantasy", "player_id"])
}

schema_data = {
    "definitions": {k: [i.value for i in v] for k, v in ENUM_MAP.items()},
    "columns": cleaned_dict,
    "coerce": True
}

with open('schema.json', 'w') as f:
    json.dump(schema_data, f, indent=4)