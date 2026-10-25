import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


def create_train_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_frac: float = 0.95,
    device: torch.device | str = "cpu",
    flatten_target: bool = False,
    group_col: str | None = None,
    test_mask: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Splits DataFrame into PyTorch train and test tensors (unscaled; PipelineDriver scales).

    Args:
        flatten_target: True gives (N,) long targets for classification;
                        False gives (N, 1) float targets for regression.
        group_col: If given, rows sharing a value (e.g. a game) stay on the same
                   side of the split. Otherwise each row is its own group.
        test_mask: If given, these rows are the test split (e.g. a held-out season)
                   and train_frac is ignored.

    Returns:
        X_train, y_train, X_test, y_test, groups_train
    """
    groups = df[group_col].to_numpy() if group_col else np.arange(len(df))
    if test_mask is not None:
        test_mask = np.asarray(test_mask, dtype=bool)
        train_idx, test_idx = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
    else:
        splitter = GroupShuffleSplit(n_splits=1, train_size=train_frac)
        train_idx, test_idx = next(splitter.split(np.zeros(len(df)), groups=groups))

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32, device=device)
    y = torch.tensor(df[target_col].values, device=device)
    y = y.long() if flatten_target else y.float().unsqueeze(-1)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], groups[train_idx]


def generate_class_weights(data: torch.Tensor) -> torch.Tensor:
    data_np = data.detach().cpu().numpy().astype(int)
    classes = np.unique(data_np)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=data_np)
    return torch.tensor(weights, dtype=torch.float32)
