import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Dict, Any, Optional

class PenaltyPredictor:
    def __init__(
        self,
        possible_penalties: List[Any],
        feature_cols: List[str],
        categorical_cols: List[str],
        lgb_params: Optional[Dict[str, Any]] = None
    ):
        self.possible_penalties = possible_penalties
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.lgb_params = lgb_params or {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
            "random_state": 42
        }
        # Dictionary storing all N trained LightGBM models
        self.models: Dict[Any, lgb.LGBMClassifier] = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures categorical columns are typed for LightGBM."""
        available_columns = [col for col in self.feature_cols if col in df.columns]
        X = df[available_columns].copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    def fit(self, df: pd.DataFrame, target_prefix: str = "has_") -> "PenaltyPredictor":
        """
        Trains N binary classifiers across all possible_penalties.
        Assumes target columns exist as f"{target_prefix}{penalty.name.lower()}"
        """
        X = self._prepare_features(df)

        for penalty in self.possible_penalties:
            target_col = f"{target_prefix}{penalty.name.lower()}"
            if target_col not in df.columns:
                raise KeyError(f"Expected target column '{target_col}' in DataFrame.")

            y = df[target_col].astype(int)

            # Calculate class imbalance weighting
            num_pos = y.sum()
            num_neg = len(y) - num_pos
            pos_weight = (num_neg / max(num_pos, 1)) if num_pos > 0 else 1.0

            clf = lgb.LGBMClassifier(
                **self.lgb_params,
                scale_pos_weight=min(pos_weight, 10.0) # cap extreme weights
            )
            clf.fit(X, y)
            self.models[penalty] = clf

        return self

    def predict_probs(self, scenario_and_play: pd.DataFrame) -> Dict[Any, float]:
        """Calculates P(Penalty) for all N models given a scenario and play call."""
        X = self._prepare_features(scenario_and_play)
        return {
            penalty: float(model.predict_proba(X)[0, 1])
            for penalty, model in self.models.items()
        }

    def sample_penalties(self, scenario_and_play: pd.DataFrame) -> List[Any]:
        """Bernoulli sampling across all N penalties for simulator execution."""
        probs = self.predict_probs(scenario_and_play)
        return [
            penalty for penalty, prob in probs.items() 
            if np.random.rand() < prob
        ]

    def save(self, filepath: str) -> None:
        """Saves the predictor instance and all N trained models into ONE file."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PenaltyPredictor":
        """Loads the predictor instance and all N models from a single file."""
        return joblib.load(filepath)