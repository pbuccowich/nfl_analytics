from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import torch

from core.trainer.ensemble import EnsembleModel
from nfl.data_management.data_manager import DataManager
from nfl.lib.constants import PLAY_COLS, SCENARIO_COLS
from nfl.lib.datamodels import Scenario
from nfl.lib.nfl_enums import PassType, PlayType, RunGaps, RunLocations


@dataclass(frozen=True)
class PlayCall:
    """Everything a coach chooses before the snap."""

    play_type: PlayType
    pass_length: PassType | None = None
    pass_location: RunLocations | None = None
    run_location: RunLocations | None = None
    run_gap: RunGaps | None = None
    shotgun: bool = False
    no_huddle: bool = False

    def to_raw(self) -> dict:
        """Play columns as they look in the raw data, before DataManager cleaning."""
        is_pass = self.play_type == PlayType.PASS
        return {
            "play_type": self.play_type,
            "pass_length": self.pass_length,
            "pass_location": self.pass_location,
            "run_location": self.run_location,
            "run_gap": self.run_gap,
            "shotgun": float(self.shotgun),
            "no_huddle": float(self.no_huddle),
            "qb_dropback": float(is_pass),
            "qb_spike": float(self.play_type == PlayType.QB_SPIKE),
            "qb_kneel": 0.0,
        }

    def __str__(self) -> str:
        parts = [
            self.play_type,
            self.pass_length,
            self.pass_location,
            self.run_location,
            self.run_gap,
            "shotgun" if self.shotgun else None,
            "no huddle" if self.no_huddle else None,
        ]
        return " ".join(str(p) for p in parts if p)


def all_play_calls() -> list[PlayCall]:
    """Every valid call: passes by depth and direction, runs by direction and gap
    (middle runs have no gap in the data), each with or without shotgun and no-huddle,
    plus punt, field goal and spike."""
    formations = list(product([False, True], repeat=2))
    passes = [
        PlayCall(
            PlayType.PASS,
            pass_length=length,
            pass_location=loc,
            shotgun=sg,
            no_huddle=nh,
        )
        for length, loc, (sg, nh) in product(PassType, RunLocations, formations)
    ]
    run_options = [(RunLocations.MIDDLE, None)] + [
        (loc, gap) for loc in (RunLocations.LEFT, RunLocations.RIGHT) for gap in RunGaps
    ]
    runs = [
        PlayCall(PlayType.RUN, run_location=loc, run_gap=gap, shotgun=sg, no_huddle=nh)
        for (loc, gap), (sg, nh) in product(run_options, formations)
    ]
    specials = [
        PlayCall(PlayType.PUNT),
        PlayCall(PlayType.FIELD_GOAL),
        PlayCall(PlayType.QB_SPIKE),
    ]
    return passes + runs + specials


class WPAPlayCaller:
    """Scores every play call with the WPA ensemble and picks the best.

    Score is mean - uncertainty_penalty * std across the ensemble. The highest raw mean
    tends to be a play the model overrates, usually one rarely called in that spot, and
    the ensemble disagrees most there, so the penalty steers away from it.
    """

    def __init__(
        self,
        ensemble: EnsembleModel,
        plays: list[PlayCall] | None = None,
        uncertainty_penalty: float = 1.0,
    ):
        feature_names = ensemble.config.get("feature_names")
        if not feature_names:
            raise ValueError("Ensemble has no saved feature_names; retrain it.")
        self.feature_names: list[str] = feature_names
        self.ensemble = ensemble
        self.plays = plays or all_play_calls()
        self.uncertainty_penalty = uncertainty_penalty

    def _features(self, scenario: Scenario) -> torch.Tensor:
        """One row per play, encoded by the same DataManager code used in training."""
        raw = pd.DataFrame([{**scenario.to_dict(), **p.to_raw()} for p in self.plays])
        df = DataManager.clean_data(raw)
        df, _ = DataManager.one_hot_encode_play_types(
            df, scenario_columns=SCENARIO_COLS, play_columns=PLAY_COLS
        )
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Can't build model features: missing {sorted(missing)}")
        X = df[self.feature_names].to_numpy(dtype=np.float32)
        device = next(self.ensemble.parameters()).device
        return torch.tensor(X, device=device)

    def rank(self, scenario: Scenario) -> pd.DataFrame:
        """Every play with predicted WPA, ensemble std and score, best first."""
        mean, std = self.ensemble(self._features(scenario))
        mean, std = mean.squeeze(-1).cpu().numpy(), std.squeeze(-1).cpu().numpy()
        return (
            pd.DataFrame(
                {
                    "play": self.plays,
                    "wpa": mean,
                    "std": std,
                    "score": mean - self.uncertainty_penalty * std,
                }
            )
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

    def select_play(self, scenario: Scenario) -> PlayCall:
        return self.rank(scenario).loc[0, "play"]
