import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Dict, Any, Union, Optional

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
        # Storage for trained LGBM models keyed by penalty name or enum
        self.models: Dict[Any, lgb.LGBMClassifier] = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure categorical columns are correctly typed for LightGBM."""
        X = df[self.feature_cols].copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    def fit(self, df: pd.DataFrame, target_col_prefix: str = "has_") -> "PenaltyPredictor":
        """
        Train a binary LightGBM classifier for each penalty in POSSIBLE_PENALTIES.
        
        Assumes target columns in df exist as: f"{target_col_prefix}{penalty.name.lower()}"
        """
        X = self._prepare_features(df)

        for penalty in self.possible_penalties:
            # Map enum or spec to target column name
            penalty_key = penalty.name.lower()
            target_col = f"{target_col_prefix}{penalty_key}"

            if target_col not in df.columns:
                raise KeyError(f"Target column '{target_col}' not found in training DataFrame.")

            y = df[target_col].astype(int)
            
            # Handle class imbalance: penalties are rare events (~0.5% - 3%)
            num_pos = y.sum()
            num_neg = len(y) - num_pos
            pos_weight = (num_neg / max(num_pos, 1)) if num_pos > 0 else 1.0
            
            # Cap extreme scale_pos_weight to avoid over-predicting rare fouls
            capped_weight = min(pos_weight, 10.0)

            model = lgb.LGBMClassifier(
                **self.lgb_params,
                scale_pos_weight=capped_weight
            )
            model.fit(X, y)
            self.models[penalty] = model

        return self

    def predict_probs(self, scenario_and_play: pd.DataFrame) -> Dict[Any, float]:
        """Returns P(Penalty) for each modeled penalty given a single scenario + play call."""
        X = self._prepare_features(scenario_and_play)
        probs = {}
        for penalty, model in self.models.items():
            # Class 1 probability
            probs[penalty] = float(model.predict_proba(X)[0, 1])
        return probs

    def sample_penalties(self, scenario_and_play: pd.DataFrame) -> List[Any]:
        """Runs Bernoulli sampling across all penalties for simulator step execution."""
        probs = self.predict_probs(scenario_and_play)
        triggered = []
        for penalty, prob in probs.items():
            if np.random.rand() < prob:
                triggered.append(penalty)
        return triggered

    def save(self, filepath: str) -> None:
        """Serialize the fitted class object to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PenaltyPredictor":
        """Load a pre-trained PenaltyPredictor instance from disk."""
        return joblib.load(filepath)