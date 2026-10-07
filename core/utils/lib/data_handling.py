import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_train_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_frac: float = 0.95,
    device: torch.device | str = "cpu",
    flatten_target: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
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

    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    X = df[feature_cols].values
    y = df[target_col].values

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, train_size=train_frac, shuffle=True
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Convert features to float tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

    # Determine target tensor type and shape
    if flatten_target:
        # 1D long integer tensors for classification (CrossEntropyLoss)
        y_train = torch.tensor(y_train_raw, dtype=torch.long, device=device)
        y_test = torch.tensor(y_test_raw, dtype=torch.long, device=device)
    else:
        # 2D float tensors for regression (MSELoss)
        y_train = torch.tensor(y_train_raw, dtype=torch.float32, device=device).unsqueeze(-1)
        y_test = torch.tensor(y_test_raw, dtype=torch.float32, device=device).unsqueeze(-1)

    return X_train, y_train, X_test, y_test, scaler