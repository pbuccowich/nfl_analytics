from pathlib import Path

import numpy as np
import torch

from core.trainer.pipeline import PipelineDriver
from core.trainer.solver import Solver
from core.utils.lib.data_handling import create_train_test_split
from nfl.data_management.data_manager import DataManager
from nfl.lib.constants import (
    EXCLUDED_PLAY_TYPES,
    PLAY_COLS,
    SCENARIO_COLS,
    TEST_SEASON,
)
from nfl.models.wpa_predictor import ResidualWPAPredictor, WPAPredictor


def _get_parameter_grid(input_size: int = 10, residual: bool = False) -> dict:
    """(low, high, distribution) specs are searched; plain values are fixed.

    The residual model searches blocks over the same range as the plain model's
    layers, so you can see whether it picks deeper when depth is easy to train.
    """
    depth_key = "num_blocks" if residual else "num_hidden_layers"
    return {
        "batch_size": (64, 512, "int"),
        "lr": (1e-5, 1e-2, "log"),
        depth_key: (1, 20, "int"),
        "hidden_size": (input_size, input_size * 5, "int"),
        "weight_decay": (1e-5, 1e-1, "log"),
        "dropout_rate": (0.05, 0.25, "float"),
        "relu_slope": (0.01, 0.1, "float"),
        "use_tanh_output": False,
    }


def train_wpa_predictor(
    num_epoch: int = 100,
    num_splits: int = 5,
    data=None,
    feature_cols=None,
    residual: bool = False,
    n_trials: int = 30,
):
    """Trains the plain MLP WPA ensemble, or with residual=True the ResidualMLP one
    (saved separately, so the two can be compared on the same 2025 test split)."""
    model_cls = ResidualWPAPredictor if residual else WPAPredictor
    save_path = (
        "nfl/artifacts/residual_wpa_ensemble.pth"
        if residual
        else "nfl/artifacts/wpa_ensemble.pth"
    )
    if data is None:
        data = DataManager.get_data(
            path_to_json=(Path.cwd() / "nfl/data").resolve()
        )  # Pulls data
        # maps to enums, drops unwanted data
        data = DataManager.clean_data(
            data, excluded_play_types=EXCLUDED_PLAY_TYPES, target_col="wpa"
        )
    if feature_cols is None:
        data, feature_cols = DataManager.one_hot_encode_play_types(
            data, scenario_columns=SCENARIO_COLS, play_columns=PLAY_COLS
        )  # one hot encode play_type and extract columns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda device count: ", torch.cuda.device_count())
    print(f"Using {device} device")

    param_grid = _get_parameter_grid(input_size=len(feature_cols), residual=residual)
    split = create_train_test_split(
        df=data,
        feature_cols=feature_cols,
        target_col="wpa",
        test_mask=(data["season"] == TEST_SEASON).to_numpy(),
        device=device,
        group_col="game_id",
    )
    X_train, y_train, X_test, y_test, groups_train = split

    # 1. Initialize Driver
    pipeline = PipelineDriver(
        solver_cls=Solver,
        model_cls=model_cls,
        criterion=torch.nn.MSELoss().to(device),
        device=device,
        n_splits=num_splits,
        num_epochs=num_epoch,
        max_delta_threshold=0.15,
    )

    # 2. Select parameters & check validation delta
    _best_hparams = pipeline.select_parameters(
        X=X_train,
        y=y_train,
        param_space=param_grid,
        n_trials=n_trials,
        groups=groups_train,
        search_frac=0.3,
    )

    # 3. Train the K-model Ensemble
    ensemble = pipeline.train_networks(
        X=X_train,
        y=y_train,
        save_path=save_path,
        groups=groups_train,
        feature_names=feature_cols,
    )

    # 4. Predict
    mean_predictions, uncertainty = ensemble(X_test)

    y_true = y_test.detach().cpu().numpy().flatten()
    y_pred = mean_predictions.detach().cpu().numpy().flatten()

    mae = np.mean(np.abs(y_true - y_pred))

    # 2. Root Mean Squared Error (RMSE) - Penalizes larger errors/outliers more heavily
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # 3. Mean Absolute Percentage Error (MAPE) - Relative scale (% error)
    # Note: Add small epsilon to denominator to avoid division by zero if target has exact zeros
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100

    print(f"{model_cls.__name__}, held-out {TEST_SEASON} season:")
    mse, mse_zero = np.mean((y_true - y_pred) ** 2), np.mean(y_true**2)
    print(f"MSE:  {mse:.6f} (predict 0: {mse_zero:.6f})")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return rmse, ensemble
