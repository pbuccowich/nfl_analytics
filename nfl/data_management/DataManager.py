from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from nfl.lib import enums
from collections import defaultdict

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
    "roof": enums.RoofType
}

class DataManager():
    # Load configuration once at the class level when the module imports
    _config_file = Path("nfl/data_management/config.yaml")
    if _config_file.exists():
        with open(_config_file, "r") as f:
            _raw_config = yaml.safe_load(f)
    else:
        print("didn't find the config file!")
        _raw_config = {}

    # Expose the configuration lists to all class methods
    bool_columns = _raw_config.get("bool_columns", [])
    drop_columns = _raw_config.get("drop_columns", [])
    yardline_columns = _raw_config.get("yardline_columns", [])
    date_columns = _raw_config.get("date_columns", [])

    @classmethod
    def get_data(cls, path_to_json: Path) -> pd.DataFrame:
        all_df = pd.DataFrame()
        path = Path(path_to_json)
        for f in path.glob('*.csv'):
            print(f"reading {f}")
            _df = pd.read_csv(f, 
                             low_memory=False,
                             parse_dates=cls.date_columns)
            all_df = pd.concat([all_df, _df], ignore_index=True)

        return cls.clean_data(all_df)

    @classmethod
    def _normalize_to_boolean(cls, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        target_cols = [c for c in columns if c in df.columns]
        for col in target_cols:
            df[col] = df[col].astype('boolean')
        return df

    @classmethod
    def _drop_columns(cls, df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
        matched_columns = [
            col for col in df.columns 
            if any(sub in col for sub in cols_to_drop)
        ]
        return df.drop(columns=matched_columns)

    @classmethod
    def _normalize_yardlines(cls, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna("").astype(str).str.strip()
            raw_num = df[col].str.split().str[-1]
            yard_num = pd.to_numeric(raw_num, errors='coerce').fillna(0)
            team_prefix = df[col].str.split().str[0]
            
            is_own_side = (team_prefix == df['posteam'])
            is_fifty = (team_prefix == "50")
            
            normalized = np.where(
                is_fifty, 50,
                np.where(is_own_side, yard_num, 100 - yard_num)
            )
            df[col] = normalized
            
        return df

    @classmethod
    def _get_day_of_season(cls, date_col: pd.Series) -> pd.Series:
        dates = pd.to_datetime(date_col)
        baseline = pd.to_datetime(dates.dt.year.astype(str) + '-08-01')
        return (dates - baseline).dt.days

    @classmethod
    def _cast_to_enums(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, enum_class in ENUM_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: enum_class(x) if pd.notnull(x) else x)

        return df
    
    @classmethod
    def clean_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        corrected_bool = cls._normalize_to_boolean(data, cls.bool_columns)
        normalized_yardlines = cls._normalize_yardlines(corrected_bool, cls.yardline_columns)
        
        # Pull the primary date column from configuration
        date_col_name = cls.date_columns[0] if cls.date_columns else 'game_date'
        normalized_yardlines["day_of_season"] = cls._get_day_of_season(normalized_yardlines[date_col_name])
        
        dropped_columns = cls._drop_columns(normalized_yardlines, cls.drop_columns)
        switch_to_enums = cls._cast_to_enums(dropped_columns)

        return switch_to_enums