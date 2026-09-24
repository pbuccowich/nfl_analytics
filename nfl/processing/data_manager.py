from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from nfl.lib import enums

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
    "roof": enums.RoofType,
}


class DataManager:
    _config_file = Path("nfl/data_management/config.yaml")
    if _config_file.exists():
        with open(_config_file, "r") as f:
            _raw_config = yaml.safe_load(f)
    else:
        _raw_config = {}

    bool_columns = _raw_config.get("bool_columns", [])
    drop_columns = _raw_config.get("drop_columns", [])
    yardline_columns = _raw_config.get("yardline_columns", [])
    date_columns = _raw_config.get("date_columns", [])

    @classmethod
    def get_data(cls, path_to_json: Path) -> pd.DataFrame:
        """Reads CSV files and applies base structural normalizations."""
        all_df = pd.DataFrame()
        path = Path(path_to_json)

        for f in path.glob("*.csv"):
            _df = pd.read_csv(
                f, low_memory=False, parse_dates=cls.date_columns
            )
            all_df = pd.concat([all_df, _df], ignore_index=True)

        df = cls._normalize_to_boolean(all_df, cls.bool_columns)
        df = cls._normalize_yardlines(df, cls.yardline_columns)

        date_col_name = (
            cls.date_columns[0] if cls.date_columns else "game_date"
        )
        if date_col_name in df.columns:
            df["day_of_season"] = cls._get_day_of_season(df[date_col_name])

        df = cls._drop_columns(df, cls.drop_columns)
        return cls._cast_to_enums(df)

    @classmethod
    def clean_data(
        cls,
        df: pd.DataFrame,
        excluded_play_types: list | set | None = None,
        target_col: str = "epa",
        fill_na_value: float = 0.0,
    ) -> pd.DataFrame:
        """Filters invalid records, derives play indicators, and handles missing values."""
        df = df.copy()

        # 1. Drop missing targets
        if target_col in df.columns:
            df = df.dropna(subset=[target_col])

        # 2. Filter excluded play types
        if excluded_play_types and "play_type" in df.columns:
            df = df[
                df["play_type"].notna()
                & ~df["play_type"].isin(excluded_play_types)
            ]

        # 3. Derive venue indicators & drop raw source columns
        if "roof" in df.columns:
            df["has_roof"] = df["roof"].isin(
                [enums.RoofType.DOME, enums.RoofType.CLOSED]
            )
            df = df.drop(columns=["roof"])

        if "surface" in df.columns:
            df["has_turf"] = df["surface"] != enums.SurfaceType.GRASS
            df = df.drop(columns=["surface"])

        # 4. Map positional gaps
        if "run_gap" in df.columns:
            gap_map = {"guard": 1, "tackle": 2, "end": 3}
            df["run_gap"] = (
                df["run_gap"].map(gap_map).fillna(-5).astype(int)
            )

        # 5. Convert boolean columns to float before filling NAs
        bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns
        df[bool_cols] = df[bool_cols].astype(float)

        return df.fillna(fill_na_value)

    @classmethod
    def prepare_features(
        cls,
        df: pd.DataFrame,
        scenario_columns: list[str],
        play_columns: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        """Encodes categorical features and outputs final feature set columns."""
        df = df.copy()

        # One-hot encode play types
        if "play_type" in df.columns:
            df = pd.get_dummies(
                df, columns=["play_type"], dtype=float, prefix="play_type"
            )

        # Assemble target feature columns
        epa_feature_columns = [
            col
            for col in df.columns
            if (
                col in scenario_columns
                or col in play_columns
                or col.startswith("play_type_")
            )
            and col not in {"play_type", "play_type_nfl"}
        ]

        # Convert feature subset to numeric & apply zero-fill safeguard
        df[epa_feature_columns] = (
            df[epa_feature_columns].astype(float).fillna(0.0)
        )

        return df, epa_feature_columns

    # --- Internal Helper Methods ---

    @classmethod
    def _normalize_to_boolean(
        cls, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        target_cols = [c for c in columns if c in df.columns]
        for col in target_cols:
            df[col] = df[col].astype("boolean")
        return df

    @classmethod
    def _drop_columns(
        cls, df: pd.DataFrame, cols_to_drop: list[str]
    ) -> pd.DataFrame:
        matched_columns = [
            col
            for col in df.columns
            if any(sub in col for sub in cols_to_drop)
        ]
        return df.drop(columns=matched_columns)

    @classmethod
    def _normalize_yardlines(
        cls, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna("").astype(str).str.strip()
            raw_num = df[col].str.split().str[-1]
            yard_num = pd.to_numeric(raw_num, errors="coerce").fillna(0)
            team_prefix = df[col].str.split().str[0]

            is_own_side = team_prefix == df["posteam"]
            is_fifty = team_prefix == "50"

            df[col] = np.where(
                is_fifty,
                50,
                np.where(is_own_side, yard_num, 100 - yard_num),
            )
        return df

    @classmethod
    def _get_day_of_season(cls, date_col: pd.Series) -> pd.Series:
        dates = pd.to_datetime(date_col)
        baseline = pd.to_datetime(dates.dt.year.astype(str) + "-08-01")
        return (dates - baseline).dt.days

    @classmethod
    def _cast_to_enums(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, enum_class in ENUM_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: enum_class(x) if pd.notnull(x) else x
                )
        return df