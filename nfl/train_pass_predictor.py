from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss

from core.trainer.pipeline import PipelineDriver
from core.trainer.solver import Solver
from core.utils.lib.data_handling import create_train_test_split, generate_class_weights
from nfl.data_management.data_manager import DataManager
from nfl.lib.constants import (
    EXCLUDED_PLAY_TYPES,
    PASS_YARDS_OUTCOMES,
    PASS_YARDS_RANGE,
    PLAY_COLS,
    SCENARIO_COLS,
)
from nfl.models.play_outcome_engine import PassDistanceEngine, PassOutcomeEngine


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


def _prepare_pass_data(data, feature_cols):
    """Loads and cleans data if not given, keeps regular-down dropbacks without a
    penalty, adds outcome targets and one-hot encodes the play columns."""
    if data is None:
        data = DataManager.get_data(path_to_json=(Path.cwd() / "nfl/data").resolve())
        data = DataManager.clean_data(
            data,
            excluded_play_types=EXCLUDED_PLAY_TYPES,
            target_col="epa",
        )

    data = DataManager.select_pass_plays(data)
    data = DataManager.create_pass_outcome_targets(data)

    if feature_cols is None:
        data, feature_cols = DataManager.one_hot_encode_play_types(
            data,
            scenario_columns=SCENARIO_COLS,
            play_columns=PLAY_COLS,
        )
    return data, feature_cols


def train_pass_outcome_engine(
    num_epoch: int = 100,
    num_splits: int = 5,
    data=None,
    feature_cols=None,
    weigh_classes: bool = False,
    n_trials: int = 30,
):
    data, feature_cols = _prepare_pass_data(data, feature_cols)

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
        n_trials=n_trials,
        groups=groups_train,
        search_frac=0.3,
    )

    # 3. Train the K-model Ensemble
    ensemble = pipeline.train_networks(
        X=X_train,
        y=y_train,
        save_path="nfl/artifacts/pass_outcome_ensemble.pth",
        groups=groups_train,
        feature_names=feature_cols,
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


def train_pass_yards_engine(
    num_epoch: int = 100,
    num_splits: int = 5,
    data=None,
    feature_cols=None,
    n_trials: int = 30,
):
    """Trains P(yards | scenario, call, outcome) for completions, scrambles and sacks.

    The outcome is a model input, so the simulator samples the outcome first and
    then the yards given that outcome.
    """
    data, feature_cols = _prepare_pass_data(data, feature_cols)
    data, outcome_cols = DataManager.create_pass_yards_targets(data)
    feature_cols = feature_cols + outcome_cols

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    X_train, y_train, X_test, y_test, groups_train = create_train_test_split(
        df=data,
        feature_cols=feature_cols,
        target_col="pass_yards_target",
        train_frac=0.95,
        device=device,
        flatten_target=True,
        group_col="game_id",
    )

    pipeline = PipelineDriver(
        solver_cls=Solver,
        model_cls=PassDistanceEngine,
        criterion=torch.nn.CrossEntropyLoss().to(device),
        device=device,
        n_splits=num_splits,
        num_epochs=num_epoch,
        max_delta_threshold=0.15,
    )
    pipeline.select_parameters(
        X=X_train,
        y=y_train,
        param_space=_get_parameter_grid(input_size=len(feature_cols)),
        n_trials=n_trials,
        groups=groups_train,
        search_frac=0.3,
    )
    ensemble = pipeline.train_networks(
        X=X_train,
        y=y_train,
        save_path="nfl/artifacts/pass_yards_ensemble.pth",
        groups=groups_train,
        feature_names=feature_cols,
    )

    ce_loss = _report_yards_metrics(
        ensemble, X_train, y_train, X_test, y_test, feature_cols, outcome_cols
    )
    return ce_loss, ensemble


def _report_yards_metrics(
    ensemble, X_train, y_train, X_test, y_test, feature_cols, outcome_cols
) -> float:
    """Held-out test metrics, against a baseline that predicts each outcome's yards
    histogram from the train split (so it knows the outcome but not the situation).
    Both get the same field mask, so the comparison is fair."""
    n_bins = PASS_YARDS_RANGE[1] - PASS_YARDS_RANGE[0] + 1
    yards = torch.arange(
        PASS_YARDS_RANGE[0], PASS_YARDS_RANGE[1] + 1, device=X_test.device
    )
    yardline = X_test[:, feature_cols.index("yardline_100")]
    outcome_idx = [feature_cols.index(c) for c in outcome_cols]
    test_outcome = X_test[:, outcome_idx].argmax(dim=1)
    train_outcome = X_train[:, outcome_idx].argmax(dim=1)

    logits, _ = ensemble(X_test)
    model_probs = PassDistanceEngine.yards_proba(logits, yardline)

    # Half-count smoothing so bins never seen in train don't get zero probability
    hist = torch.stack(
        [
            torch.bincount(y_train[train_outcome == i], minlength=n_bins) + 0.5
            for i in range(len(outcome_cols))
        ]
    ).float()
    base_probs = PassDistanceEngine.yards_proba(hist.log()[test_outcome], yardline)

    def scores(probs, mask):
        p, y = probs[mask], y_test[mask]
        ce = -p.gather(1, y.unsqueeze(1)).clamp_min(1e-12).log().mean()
        # CRPS: squared gap between predicted and actual CDFs, summed over yards.
        # Unlike cross-entropy it gives credit for being close to the right yard.
        step = (torch.arange(n_bins, device=y.device) >= y.unsqueeze(1)).float()
        crps = ((p.cumsum(dim=1) - step) ** 2).sum(dim=1).mean()
        mean_yards = (p * yards).sum(dim=1).mean()
        p_td = p[torch.arange(len(y)), yardline[mask].long() - PASS_YARDS_RANGE[0]]
        return ce.item(), crps.item(), mean_yards.item(), p_td.mean().item()

    print(
        f"{'outcome':<10}{'n':>7}  {'CE model/base':>15}  {'CRPS model/base':>17}"
        f"  {'mean yds pred/act':>18}  {'P(TD) pred/act':>16}"
    )
    groups = [(o.value, test_outcome == i) for i, o in enumerate(PASS_YARDS_OUTCOMES)]
    groups.append(("all", torch.ones_like(y_test, dtype=torch.bool)))
    for name, mask in groups:
        ce_m, crps_m, yds_m, td_m = scores(model_probs, mask)
        ce_b, crps_b, _, _ = scores(base_probs, mask)
        actual = yards[y_test[mask]]
        td_act = (actual == yardline[mask]).float().mean().item()
        print(
            f"{name:<10}{int(mask.sum()):>7}  {ce_m:>7.3f}/{ce_b:<7.3f}  "
            f"{crps_m:>8.3f}/{crps_b:<8.3f}  {yds_m:>8.2f}/{actual.float().mean():<9.2f}"  # noqa: E501
            f"  {td_m:>7.4f}/{td_act:<8.4f}"
        )
    return scores(model_probs, groups[-1][1])[0]
