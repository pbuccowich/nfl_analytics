import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def create_train_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_frac: float = 0.95,
    device: torch.device | str = "cpu",
    flatten_target: bool = False,
    group_col: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray | None]:
    """
    Splits DataFrame into scaled PyTorch train and test tensors.

    Args:
        df: Input DataFrame containing features and targets.
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        train_frac: Fraction of data allocated to training.
        device: PyTorch device ('cpu' or 'cuda').
        flatten_target: If True, reshapes y to a 1D tensor (N,). 
                       If False, preserves 2D column matrix (N, 1).
        group_col: If given, rows sharing a value (e.g. a game) stay on the same
                   side of the split.

    Returns:
        X_train, y_train, X_test, y_test, groups_train (None without group_col)
    """
    X = df[feature_cols].values
    y = df[target_col].values

    if group_col is None:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, train_size=train_frac, shuffle=True
        )
        groups_train = None
    else:
        groups = df[group_col].values
        splitter = GroupShuffleSplit(n_splits=1, train_size=train_frac)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train_raw, y_test_raw = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

    # Scale features
#    scaler = StandardScaler()
#    X_train_scaled = scaler.fit_transform(X_train_raw)
#    X_test_scaled = scaler.transform(X_test_raw)

    # Convert features to float tensors
    X_train = torch.tensor(X_train_raw, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test_raw, dtype=torch.float32, device=device)

    # Determine target tensor type and shape
    if flatten_target:
        # 1D long integer tensors for classification (CrossEntropyLoss)
        y_train = torch.tensor(y_train_raw, dtype=torch.long, device=device)
        y_test = torch.tensor(y_test_raw, dtype=torch.long, device=device)
    else:
        # 2D float tensors for regression (MSELoss)
        y_train = torch.tensor(y_train_raw, dtype=torch.float32, device=device).unsqueeze(-1)
        y_test = torch.tensor(y_test_raw, dtype=torch.float32, device=device).unsqueeze(-1)

    return X_train, y_train, X_test, y_test, groups_train

def generate_class_weights(data: torch.Tensor) -> torch.Tensor:
    data_np = data.detach().cpu().numpy().astype(int)
    classes = np.unique(data_np)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=data_np)
    return torch.tensor(weights, dtype=torch.float32)