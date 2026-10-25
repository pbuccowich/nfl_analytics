from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nfl.lib import nfl_enums as enums
from nfl.lib.constants import (
    CATEGORICAL_PLAY_COLS,
    PASS_YARDS_OUTCOMES,
    PENALTY_ENUM_MAPPING,
    YARDS_RANGE,
)

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
    keep_columns = _raw_config.get("keep_columns", [])
    yardline_columns = _raw_config.get("yardline_columns", [])
    date_columns = _raw_config.get("date_columns", [])

    @classmethod
    def get_data(cls, path_to_json: Path) -> pd.DataFrame:
        """Reads CSV files and applies base structural normalizations."""
        all_df = pd.concat(
            [
                pd.read_csv(f, low_memory=False, parse_dates=cls.date_columns)
                for f in Path(path_to_json).glob("*.csv")
            ],
            ignore_index=True,
        )

        df = cls._normalize_to_boolean(all_df, cls.bool_columns)
        df = cls._normalize_yardlines(df, cls.yardline_columns)

        date_col_name = cls.date_columns[0] if cls.date_columns else "game_date"
        if date_col_name in df.columns:
            df["day_of_season"] = cls._get_day_of_season(df[date_col_name])

        df = cls._describe_nullified_calls(df)
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
                df["play_type"].notna() & ~df["play_type"].isin(excluded_play_types)
            ]

        df = cls._describe_dropbacks_as_pass_calls(df)

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
            df["run_gap"] = df["run_gap"].map(gap_map).fillna(-5).astype(int)

        # 5. Map pass length to integer (1 for short, 2 for deep, 0 for missing/other)
        if "pass_length" in df.columns:
            df["pass_length"] = cls._map_pass_length(df["pass_length"])

        # 6. Convert boolean columns to float before filling NAs
        bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns
        df[bool_cols] = df[bool_cols].astype(float)

        return df.fillna(fill_na_value)

    @classmethod
    def select_pass_plays(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Dropbacks that are regular downs with no penalty, cleaned (NaN filled).

        The simulator runs the play as if no penalty happened and applies penalties
        separately, so penalty plays would count them twice. Two-point tries are
        exactly the pass/run plays with no down, which clean_data fills with 0.
        """
        return df[(df["pass"] == 1) & (df["penalty"] == 0) & (df["down"] >= 1)]

    @classmethod
    def create_pass_outcome_targets(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Derives hierarchical PassOutcome enum string targets and integer target indices."""
        df = df.copy()

        # Priority conditions; a scramble isn't a throw, so without this it would
        # fall through to incomplete
        conditions = [
            df.get("interception", 0) == 1,
            df.get("sack", 0) == 1,
            df.get("fumble_lost", 0) == 1,
            df.get("qb_scramble", 0) == 1,
            df.get("complete_pass", 0) == 1,
        ]

        choices = [
            enums.PassOutcome.INTERCEPTION.value,
            enums.PassOutcome.SACK.value,
            enums.PassOutcome.FUMBLE.value,
            enums.PassOutcome.SCRAMBLE.value,
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
        df["pass_outcome_target"] = (
            df["pass_outcome_str"].map(outcome_to_idx).astype(int)
        )

        return df

    @classmethod
    def create_pass_yards_targets(
        cls, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Keeps plays whose outcome needs yards and bins yards_gained.

        Needs create_pass_outcome_targets first. The outcome is an input to the yards
        model (one-hot). Adds pass_yards_bin (yards - low, clipped to
        YARDS_RANGE) and two flags for yards that are only a bound:
        pass_yards_at_least for touchdowns (the play stops at the goal line, so it
        might have gone farther) and pass_yards_at_most for safeties and losses past
        the range. Returns the frame and the outcome feature columns.
        """
        outcomes = [o.value for o in PASS_YARDS_OUTCOMES]
        df = df[df["pass_outcome_str"].isin(outcomes)].copy()

        outcome_cols = [f"pass_outcome_{o}" for o in outcomes]
        for col, outcome in zip(outcome_cols, outcomes):
            df[col] = (df["pass_outcome_str"] == outcome).astype(float)

        low, high = YARDS_RANGE
        yards = df["yards_gained"].round()
        df["pass_yards_bin"] = yards.clip(low, high).astype(int) - low
        df["pass_yards_at_least"] = (yards >= df["yardline_100"]).astype(int)
        df["pass_yards_at_most"] = (
            (yards <= df["yardline_100"] - 100) | (yards < low)
        ).astype(int)
        return df, outcome_cols

    @classmethod
    def select_run_plays(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Designed runs on a regular down with no penalty, cleaned (NaN filled).

        Same rules as select_pass_plays. Scrambles aren't here: clean_data makes
        every dropback a pass call. Kneels are their own play type.
        """
        return df[
            (df["play_type"] == enums.PlayType.RUN)
            & (df["penalty"] == 0)
            & (df["down"] >= 1)
        ]

    @classmethod
    def create_run_targets(cls, df: pd.DataFrame, fumble_class: int) -> pd.DataFrame:
        """Adds run_class (yards - low, clipped to YARDS_RANGE, or fumble_class for a
        lost fumble) and the same bound flags as create_pass_yards_targets:
        run_yards_at_least for touchdowns, run_yards_at_most for safeties and losses
        past the range. A lost fumble has no yards, so neither flag is set on it.
        """
        df = df.copy()
        low, high = YARDS_RANGE
        yards = df["yards_gained"].round()
        lost = df["fumble_lost"] == 1
        df["run_class"] = (yards.clip(low, high).astype(int) - low).where(
            ~lost, fumble_class
        )
        df["run_yards_at_least"] = ((yards >= df["yardline_100"]) & ~lost).astype(int)
        df["run_yards_at_most"] = (
            ((yards <= df["yardline_100"] - 100) | (yards < low)) & ~lost
        ).astype(int)
        return df

    @classmethod
    def create_penalty_targets(
        cls,
        df: pd.DataFrame,
        pre_snap_penalties: list[tuple[enums.PenaltyType, bool]],
        live_ball_penalties: list[tuple[enums.PenaltyType, bool]],
    ) -> pd.DataFrame:
        """Selects regular-down snaps and labels the penalty on each.

        Takes get_data output, not clean_data's, which drops the no_play rows that
        hold most fouls. Rows are every pass or run snap, including ones a foul
        wiped out (their call is restored as a pass or run), plus pre-snap fouls,
        where no play ran. Adds is_pre_snap_foul and two targets: 0 for no penalty,
        1..K for the listed (penalty, on_offense) classes, then K+1 for other
        offensive and K+2 for other defensive. For the pre-snap target a live-ball
        foul counts as no penalty; the live model doesn't use pre-snap rows.
        """
        called = df["pass"].fillna(False).astype(bool) | df["rush"].fillna(
            False
        ).astype(bool)
        flagged = df["penalty"].fillna(False).astype(bool)
        no_play = df["play_type"] == enums.PlayType.NO_PLAY
        pre_snap = no_play & ~called & flagged
        keep = (
            df["down"].notna()
            & df["play_type"].isin(
                [
                    enums.PlayType.PASS,
                    enums.PlayType.RUN,
                    enums.PlayType.NO_PLAY,
                    enums.PlayType.QB_SPIKE,
                ]
            )
            & (called | pre_snap)
            # A foul we can't pin on a side can't be labelled
            & (~flagged | df["penalty_on_offense"].notna())
        )
        df = df[keep].copy()
        pre_snap, flagged = pre_snap[keep], flagged[keep]

        nullified = (df["play_type"] == enums.PlayType.NO_PLAY) & ~pre_snap
        is_pass = df["pass"].fillna(False).astype(bool)
        df.loc[nullified & is_pass, "play_type"] = enums.PlayType.PASS
        df.loc[nullified & ~is_pass, "play_type"] = enums.PlayType.RUN
        # qb_dropback is unreliable on wiped-out plays; pass is set from the desc
        df.loc[nullified, "qb_dropback"] = is_pass[nullified]

        df["is_pre_snap_foul"] = pre_snap.astype(float)
        df["pre_snap_penalty_target"] = cls._penalty_class(
            df, pre_snap_penalties
        ).where(pre_snap, 0)
        df["live_penalty_target"] = cls._penalty_class(df, live_ball_penalties).where(
            flagged & ~pre_snap, 0
        )
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

        # One-hot encode against fixed categories (e.g. play_type_pass); values outside
        # them, like a missing location, encode as all zeros
        categorical = [c for c in CATEGORICAL_PLAY_COLS if c in df.columns]
        for col in categorical:
            df[col] = pd.Categorical(df[col], categories=CATEGORICAL_PLAY_COLS[col])
        df = pd.get_dummies(df, columns=categorical, dtype=float)

        # Assemble target feature columns
        epa_feature_columns = [
            col
            for col in df.columns
            if (
                col in scenario_columns
                or col in play_columns
                or any(col.startswith(f"{c}_") for c in CATEGORICAL_PLAY_COLS)
            )
            and col != "play_type_nfl"
        ]

        # Convert feature subset to numeric & apply zero-fill safeguard
        df[epa_feature_columns] = df[epa_feature_columns].astype(float).fillna(0.0)

        return df, epa_feature_columns

    # --- Internal Helper Methods ---

    @classmethod
    def _penalty_class(
        cls, df: pd.DataFrame, penalties: list[tuple[enums.PenaltyType, bool]]
    ) -> pd.Series:
        """Class index of each row's (penalty_type, penalty_on_offense), with
        unlisted penalties in the other-offensive / other-defensive classes."""
        index = {key: i + 1 for i, key in enumerate(penalties)}
        other_offense, other_defense = len(penalties) + 1, len(penalties) + 2
        offense = df["penalty_on_offense"].eq(True)
        classes = [
            index.get(
                (penalty, on_offense),
                other_offense if on_offense else other_defense,
            )
            for penalty, on_offense in zip(df["penalty_type"], offense)
        ]
        return pd.Series(classes, index=df.index)

    @classmethod
    def _describe_nullified_calls(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Recovers the call on plays a foul wiped out, and who committed fouls.

        Runs before drop_columns removes desc, posteam and penalty_team. nflfastR
        marks wiped-out plays no_play and leaves the play columns blank, but still
        sets pass/rush, and desc still describes the play ("pass short left",
        "left tackle"). Only blanks on no_play rows are filled, and the other models
        exclude those rows.
        """
        if {"penalty_team", "posteam"} <= set(df.columns):
            df["penalty_on_offense"] = (df["penalty_team"] == df["posteam"]).where(
                df["penalty_team"].notna()
            )
        if not {"desc", "play_type", "pass", "rush"} <= set(df.columns):
            return df

        nullified = df["play_type"] == "no_play"
        # The text before the first penalty describes the play itself
        play_text = (
            df.loc[nullified, "desc"]
            .fillna("")
            .str.split(r"(?i)penalty", n=1, regex=True)
            .str[0]
        )
        is_pass = df.loc[nullified, "pass"].fillna(False).astype(bool)
        is_run = df.loc[nullified, "rush"].fillna(False).astype(bool) & ~is_pass
        throw = play_text.str.extract(r"\b(short|deep) (left|middle|right)\b")
        run = play_text.str.extract(r"\b(left|right) (end|tackle|guard)\b")
        up_the_middle = play_text.str.contains(r"\bup the middle\b")
        fills = {
            "pass_length": throw[0].where(is_pass),
            "pass_location": throw[1].where(is_pass),
            "run_location": run[0].where(~up_the_middle, "middle").where(is_run),
            "run_gap": run[1].where(is_run),
        }
        for col, values in fills.items():
            if col in df.columns:
                df.loc[nullified, col] = df.loc[nullified, col].fillna(values)
        return df

    @classmethod
    def _describe_dropbacks_as_pass_calls(
        cls, df: pd.DataFrame, seed: int = 0
    ) -> pd.DataFrame:
        """Makes play columns describe the call, not what happened.

        Every dropback was called as a pass, so scrambles (labelled run) become passes
        and lose their run direction. Sacks and scrambles have no pass depth or
        direction, and leaving those blank would tell the model "no throw happened".
        Instead they get the depth and direction of a randomly drawn thrown pass, so
        their outcomes count against pass calls in general.
        """
        needed = {"qb_dropback", "play_type", "pass_length", "pass_location"}
        if not needed <= set(df.columns):
            return df
        df = df.copy()
        dropback = (pd.to_numeric(df["qb_dropback"], errors="coerce") == 1) & df[
            "play_type"
        ].isin([enums.PlayType.PASS, enums.PlayType.RUN])
        df.loc[dropback, "play_type"] = enums.PlayType.PASS
        for col in ("run_location", "run_gap"):
            if col in df.columns:
                df.loc[dropback, col] = None

        throw_cols = ["pass_length", "pass_location"]
        thrown = dropback & df[throw_cols].notna().all(axis=1)
        no_throw = dropback & ~thrown
        if no_throw.any() and thrown.any():
            rng = np.random.default_rng(seed)
            donors = rng.choice(np.flatnonzero(thrown), size=no_throw.sum())
            df.loc[no_throw, throw_cols] = df[throw_cols].iloc[donors].to_numpy()
        return df

    @classmethod
    def _map_pass_length(cls, series: pd.Series) -> pd.Series:
        """Maps pass_length enum or string values to 1 (short), 2 (deep), or 0."""
        text = series.astype(str).str.lower()
        return pd.Series(
            np.select(
                [text.str.contains("short"), text.str.contains("deep")], [1, 2], 0
            ),
            index=series.index,
        )

    @classmethod
    def _normalize_to_boolean(
        cls, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        target_cols = [c for c in columns if c in df.columns]
        for col in target_cols:
            df[col] = df[col].astype("boolean")
        return df

    @classmethod
    def _drop_columns(cls, df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
        matched_columns = [
            col
            for col in df.columns
            if col not in cls.keep_columns and any(sub in col for sub in cols_to_drop)
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
                    df[col] = df[col].map(
                        lambda x: converter(x) if pd.notnull(x) else x
                    )
        return df
