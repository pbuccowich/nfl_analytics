import torch
import numpy as np
from pathlib import Path
from nfl.data_management import nfl_enums as enums
from nfl.data_management.DataManager import DataManager
from nfl.models.wpa_predictor import WPAPredictor
from core.trainer.solver import Solver
from core.trainer.pipeline import PipelineDriver
from core.utils.lib.data_handling import create_train_test_split

SCENARIO_COLS = [
    "yardline_100",
    "game_seconds_remaining",
    "has_turf",
    "temp",
    "wind",
    "has_roof",
    "ydstogo",
    "goal_to_go",
    "score_differential",
    "down",
    "div_game",
    "day_of_season",
    "series"
]

PLAY_COLS = [
    "play_type",
    "pass_location",
#    "pass_length",
#    "run_location",
    "run_gap",
    "shotgun",
    "no_huddle",
    "qb_kneel",
    "qb_spike",
    "qb_scramble",
    "air_yards"
]

RESULT_COLS = [
    "epa",  
    "wpa",
    "success",
    "result",
    "series_success",
    "tackle_for_loss",
    "saftey",
    "yards_gained",
    "touchdown",
    "fumble",
    "complete_pass",
    "rushing_yards",
    "fumble_lost",
    "interception",
    "sack",
    "penalty_yards",
]

EXCLUDED_PLAY_TYPES = {
    enums.PlayType.KICK,
    enums.PlayType.EXTRA_POINT,
    enums.PlayType.NO_PLAY,
    enums.PlayType.GAME_START,
}

def _get_parameter_grid():
    NUM_ITER = 7

    MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS = 1, 20
    MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE = 16, 256
    MIN_LEARNING_RATE, MAX_LEARNING_RATE = 1e-5, 1e-2
    MIN_BATCH_SIZE, MAX_BATCH_SIZE = 64, 512
    MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY = 1e-5, 1e-2

    BATCH_SIZE = np.linspace(MIN_BATCH_SIZE, MAX_BATCH_SIZE, num=NUM_ITER).astype(int).tolist()
    LEARNING_RATE = np.geomspace(MIN_LEARNING_RATE, MAX_LEARNING_RATE, num=NUM_ITER).tolist()
    NUM_HIDDEN_LAYERS = np.linspace(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS, num=NUM_ITER).astype(int).tolist()
    HIDDEN_SIZE = np.linspace(MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, num=NUM_ITER).astype(int).tolist()
    WEIGHT_DECAY = np.geomspace(MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY, num=NUM_ITER).tolist()

    param_grid = {
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "weight_decay": WEIGHT_DECAY,
    }

    return param_grid

def train_wpa_predictor(num_epoch: int = 100, num_splits: int = 5):
    data = DataManager.get_data(path_to_json = (Path.cwd() / "nfl/data").resolve())
    data = DataManager.clean_data(data, excluded_play_types=EXCLUDED_PLAY_TYPES, target_col="wpa")
    data, feature_cols = DataManager.prepare_features(data, scenario_columns=SCENARIO_COLS, play_columns=PLAY_COLS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda device count: ", torch.cuda.device_count())
    print(f"Using {device} device")

    param_grid = _get_parameter_grid()
    split = create_train_test_split(df=data, feature_cols=feature_cols, target_col="wpa", train_frac = 0.95, device=device)
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
    best_hparams = pipeline.select_parameters(
        X=X_train,
        y=y_train,
        param_grid=param_grid,
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