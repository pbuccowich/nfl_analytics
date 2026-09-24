import torch
from sklearn.preprocessing import StandardScaler
from typing import NamedTuple
import pandas as pd

class DatasetSplit(NamedTuple):
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    scaler: StandardScaler


def create_train_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "epa",
    train_frac: float = 0.8,
    random_state: int = 42,
    device: torch.device | str | None = None,
) -> DatasetSplit:
    """Splits, scales, and converts a DataFrame into PyTorch tensors."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 1. Train / test split
    train_df = df.sample(frac=train_frac, random_state=random_state)
    test_df = df.drop(train_df.index)

    # 2. Scale continuous features (fit on train only)
    continuous_cols = [c for c in feature_cols if df[c].nunique() > 2]

    scaler = StandardScaler()
    X_train_np = train_df[feature_cols].copy()
    X_test_np = test_df[feature_cols].copy()

    if continuous_cols:
        X_train_np[continuous_cols] = scaler.fit_transform(
            train_df[continuous_cols]
        )
        X_test_np[continuous_cols] = scaler.transform(test_df[continuous_cols])

    # 3. Convert to PyTorch tensors
    X_train = torch.tensor(
        X_train_np.values, dtype=torch.float32, device=device
    )
    y_train = torch.tensor(
        train_df[target_col].values, dtype=torch.float32, device=device
    ).unsqueeze(1)

    X_test = torch.tensor(X_test_np.values, dtype=torch.float32, device=device)
    y_test = torch.tensor(
        test_df[target_col].values, dtype=torch.float32, device=device
    ).unsqueeze(1)

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
    )