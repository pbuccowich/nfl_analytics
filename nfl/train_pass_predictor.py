from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss

from core.trainer.pipeline import PipelineDriver
from core.trainer.solver import Solver
from core.utils.lib.data_handling import create_train_test_split, generate_class_weights
from nfl.data_management.data_manager import DataManager
from nfl.lib.constants import EXCLUDED_PLAY_TYPES, PLAY_COLS, SCENARIO_COLS
from nfl.models.play_outcome_engine import PassOutcomeEngine


def _get_parameter_grid(input_size: int = 10) -> dict:
    """(low, high, distribution) specs are searched; plain values are fixed."""
    return {
        "batch_size": (64, 512, "int"),
        "lr": (1e-5, 1e-2, "log"),
        "num_hidden_layers": (1, 20, "int"),
        "hidden_size": (input_size, input_size * 30, "int"),
        "weight_decay": (1e-5, 1e-2, "log"),
        "dropout_rate": (0.0, 0.2, "float"),
        "relu_slope": (0.01, 0.1, "float"),
    }


def train_pass_outcome_engine(
    num_epoch: int = 100,
    num_splits: int = 5,
    data=None,
    feature_cols=None,
    weigh_classes: bool = False,
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
        flatten_target=True,
        group_col="game_id",
    )
    X_train, y_train, X_test, y_test, groups_train = split

    weights = generate_class_weights(y_train).to(device) if weigh_classes else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)

    # 1. Initialize Driver with CrossEntropyLoss
    pipeline = PipelineDriver(
        solver_cls=Solver,
        model_cls=PassOutcomeEngine,
        criterion=criterion,
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
        groups=groups_train,
        search_frac=0.3,
    )

    # 3. Train the K-model Ensemble
    ensemble = pipeline.train_networks(
        X=X_train,
        y=y_train,
        save_path="pass_outcome_ensemble.pth",
        groups=groups_train,
    )

    # 4. Predict raw logits and convert to probabilities
    logits, _ = ensemble(X_test)
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    y_pred_classes = np.argmax(probs, axis=1)

    # Evaluation Metrics for Multiclass Classification
    y_test_np = y_test.cpu().numpy()

    ce_loss = log_loss(y_test_np, probs, labels=list(range(probs.shape[1])))
    acc = accuracy_score(y_test_np, y_pred_classes)
    macro_f1 = f1_score(y_test_np, y_pred_classes, average="macro")

    print(f"Cross-Entropy Loss: {ce_loss:.4f}")
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Macro F1 Score:     {macro_f1:.4f}")

    return ce_loss, ensemble