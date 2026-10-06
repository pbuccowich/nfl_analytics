from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss

from core.trainer.pipeline import PipelineDriver
from core.trainer.solver import Solver
from core.utils.lib.data_handling import create_train_test_split
from nfl.data_management.data_manager import DataManager
from nfl.lib.constants import EXCLUDED_PLAY_TYPES, PLAY_COLS, SCENARIO_COLS
from nfl.models.play_outcome_engine import PassOutcomeEngine


def _get_parameter_grid(input_size: int = 10):
    MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS = 1, 20
    MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE = input_size, input_size * 30
    MIN_LEARNING_RATE, MAX_LEARNING_RATE = 1e-5, 1e-2
    MIN_BATCH_SIZE, MAX_BATCH_SIZE = 64, 512
    MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY = 1e-5, 1e-2
    MIN_DROPOUT_RATE, MAX_DROPOUT_RATE = 0.0, 0.5
    MIN_RELU_SLOPE, MAX_RELU_SLOPE = 0.01, 0.1

    param_space = {
        "batch_size": (MIN_BATCH_SIZE, MAX_BATCH_SIZE, "int"),
        "lr": (MIN_LEARNING_RATE, MAX_LEARNING_RATE, "log"),
        "num_hidden_layers": (MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS, "int"),
        "hidden_size": (MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, "int"),
        "weight_decay": (MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY, "log"),
        "dropout_rate": (MIN_DROPOUT_RATE, MAX_DROPOUT_RATE, "float"),
        "relu_slope": (MIN_RELU_SLOPE, MAX_RELU_SLOPE, "float"),
    }

    return param_space


def train_pass_outcome_engine(
    num_epoch: int = 100,
    num_splits: int = 5,
    data=None,
    feature_cols=None,
):
    if data is None:
        data = DataManager.get_data(
            path_to_json=(Path.cwd() / "nfl/data").resolve()
        )
        data = DataManager.clean_data(
            data,
            excluded_play_types=EXCLUDED_PLAY_TYPES,
            target_col="epa",
        )

    # Derive pass outcome target indices [0..4]
    data = DataManager.create_pass_outcome_targets(data)

    if feature_cols is None:
        data, feature_cols = DataManager.one_hot_encode_play_types(
            data,
            scenario_columns=SCENARIO_COLS,
            play_columns=PLAY_COLS,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda device count: ", torch.cuda.device_count())
    print(f"Using {device} device")

    # Pass the feature count as input_size for param bounds
    param_grid = _get_parameter_grid(input_size=len(feature_cols))

    split = create_train_test_split(
        df=data,
        feature_cols=feature_cols,
        target_col="pass_outcome_target",
        train_frac=0.95,
        device=device,
    )
    X_train, y_train, X_test, y_test, scaler = split

    # Convert targets to long integers for CrossEntropyLoss
    y_train = y_train.long().squeeze()
    y_test_np = y_test.detach().cpu().numpy().flatten().astype(int)

    # 1. Initialize Driver with CrossEntropyLoss
    pipeline = PipelineDriver(
        solver_cls=Solver,
        model_cls=PassOutcomeEngine,
        criterion=torch.nn.CrossEntropyLoss().to(device),
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
        save_path="pass_outcome_ensemble.pth",
    )

    # 4. Predict raw logits and convert to probabilities
    logits, _ = ensemble(X_test)
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    y_pred_classes = np.argmax(probs, axis=1)

    # Evaluation Metrics
    ce_loss = log_loss(y_test_np, probs, labels=list(range(probs.shape[1])))
    acc = accuracy_score(y_test_np, y_pred_classes)
    macro_f1 = f1_score(y_test_np, y_pred_classes, average="macro")

    print(f"Cross-Entropy Loss: {ce_loss:.4f}")
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Macro F1 Score:     {macro_f1:.4f}")

    return ce_loss, ensemble