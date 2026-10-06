from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nfl.lib import nfl_enums as enums
from nfl.lib.constants import PENALTY_ENUM_MAPPING

DATA_MAP = {
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
    "penalty_type": PENALTY_ENUM_MAPPING,
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
        with open(_config_file) as f:
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
        target_col: str = "wpa",
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

        if "play_type" in df.columns:
            df["play_type"] = df["play_type"].astype("category").cat.codes

        # 4. Map positional gaps
        if "run_gap" in df.columns:
            gap_map = {"guard": 1, "tackle": 2, "end": 3}
            df["run_gap"] = (
                df["run_gap"].map(gap_map).fillna(-5).astype(int)
            )

        # 5. Map pass length to integer (1 for short, 2 for deep, 0 for missing/other)
        if "pass_length" in df.columns:
            df["pass_length"] = cls._map_pass_length(df["pass_length"])

        # 6. Convert boolean columns to float before filling NAs
        bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns
        df[bool_cols] = df[bool_cols].astype(float)

        return df.fillna(fill_na_value)

    @classmethod
    def create_pass_outcome_targets(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Derives hierarchical PassOutcome enum string targets and integer target indices."""
        df = df.copy()

        # Priority conditions
        conditions = [
            df.get("interception", 0) == 1,
            df.get("sack", 0) == 1,
            df.get("fumble_lost", 0) == 1,
            df.get("complete_pass", 0) == 1,
        ]

        choices = [
            enums.PassOutcome.INTERCEPTION.value,
            enums.PassOutcome.SACK.value,
            enums.PassOutcome.FUMBLE.value,
            enums.PassOutcome.COMPLETE.value,
        ]

        # Default fallback is incomplete pass
        df["pass_outcome_str"] = np.select(
            conditions, choices, default=enums.PassOutcome.INCOMPLETE.value
        )

        # Map enum strings to target indices [0, 1, 2, 3, 4]
        outcome_to_idx = {
            outcome.value: idx for idx, outcome in enumerate(enums.PassOutcome)
        }
        df["pass_outcome_target"] = df["pass_outcome_str"].map(outcome_to_idx).astype(int)

        return df

    @classmethod
    def one_hot_encode_play_types(
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
    def _map_pass_length(cls, series: pd.Series) -> pd.Series:
        """Maps pass_length enum or string values to 1 (short), 2 (deep), or 0."""
        def _to_int(val):
            val_str = str(val).lower() if pd.notnull(val) else ""
            if "short" in val_str:
                return 1
            elif "deep" in val_str:
                return 2
            return 0

        return series.apply(_to_int).astype(int)

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
        for col, converter in DATA_MAP.items():
            if col in df.columns:
                if isinstance(converter, dict):
                    df[col] = df[col].map(converter)
                else:
                    df[col] = df[col].map(lambda x: converter(x) if pd.notnull(x) else x)
        return df