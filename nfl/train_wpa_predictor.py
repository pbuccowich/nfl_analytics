from pathlib import Path

import numpy as np
import torch

from core.trainer.pipeline import PipelineDriver
from core.trainer.solver import Solver
from core.utils.lib.data_handling import create_train_test_split
from nfl.data_management.data_manager import DataManager
from nfl.lib.constants import EXCLUDED_PLAY_TYPES, PLAY_COLS, SCENARIO_COLS
from nfl.models.wpa_predictor import WPAPredictor


def _get_parameter_grid():
    MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS = 1, 20
    MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE = 16, 256
    MIN_LEARNING_RATE, MAX_LEARNING_RATE = 1e-5, 1e-2
    MIN_BATCH_SIZE, MAX_BATCH_SIZE = 64, 512
    MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY = 1e-5, 1e-2

    param_space = {
        "batch_size": (MIN_BATCH_SIZE, MAX_BATCH_SIZE, "int"),
        "lr": (MIN_LEARNING_RATE, MAX_LEARNING_RATE, "log"),
        "num_hidden_layers": (MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS, "int"),
        "hidden_size": (MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, "int"),
        "weight_decay": (MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY, "log"),
    }

    return param_space

def train_wpa_predictor(num_epoch: int = 100, num_splits: int = 5, data = None, feature_cols = None):
    if data is None:
        data = DataManager.get_data(
            path_to_json = (Path.cwd() / "nfl/data").resolve()
            ) # Pulls data
        # maps to enums, drops unwanted data
        data = DataManager.clean_data(data,
                                      excluded_play_types=EXCLUDED_PLAY_TYPES,
                                      target_col="wpa")
    if feature_cols is None:
        data, feature_cols = DataManager.one_hot_encode_play_types(data,
                                                                   scenario_columns=SCENARIO_COLS,
                                                                   play_columns=PLAY_COLS) # one hot encode play_type and extract columns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda device count: ", torch.cuda.device_count())
    print(f"Using {device} device")

    param_grid = _get_parameter_grid()
    split = create_train_test_split(df=data,
                                    feature_cols=feature_cols,
                                    target_col="wpa",
                                    train_frac = 0.95,
                                    device=device)
    X_train, y_train, X_test, y_test, scaler = split

    # 1. Initialize Driver
    pipeline = PipelineDriver(
        solver_cls=Solver,
        model_cls=WPAPredictor,
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
        n_trials=30,
    )

    # 3. Train the K-model Ensemble
    ensemble = pipeline.train_networks(
        X=X_train,
        y=y_train,
        save_path="ensemble.pth",
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

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return rmse, ensemble
