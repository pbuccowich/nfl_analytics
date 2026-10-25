from pathlib import Path

import torch
import torch.nn.functional as F

from core.trainer.pipeline import PipelineDriver
from core.trainer.solver import Solver
from core.utils.lib.data_handling import create_train_test_split
from nfl.data_management.data_manager import DataManager
from nfl.lib.constants import (
    LIVE_BALL_PENALTIES,
    PLAY_COLS,
    PRE_SNAP_PENALTIES,
    SCENARIO_COLS,
    TEST_SEASON,
)
from nfl.models.penalty_engine import (
    LiveBallPenaltyEngine,
    PenaltyEngine,
    PreSnapPenaltyEngine,
)

# No play runs on a pre-snap foul, so formation is all we know about the call
PRE_SNAP_PLAY_COLS = ["shotgun", "no_huddle"]


def _get_parameter_grid(input_size: int = 10) -> dict:
    """(low, high, distribution) specs are searched; plain values are fixed."""
    return {
        "batch_size": (256, 2048, "int"),
        "lr": (1e-5, 1e-2, "log"),
        "num_hidden_layers": (1, 6, "int"),
        "hidden_size": (input_size, input_size * 10, "int"),
        "weight_decay": (1e-5, 1e-2, "log"),
        "dropout_rate": (0.0, 0.2, "float"),
        "relu_slope": (0.01, 0.1, "float"),
    }


def train_penalty_engines(
    num_epoch: int = 100,
    num_splits: int = 5,
    raw_data=None,
    n_trials: int = 30,
):
    """Trains the pre-snap and live-ball penalty engines.

    raw_data is get_data output. clean_data's output won't do: it drops no_play
    rows, which hold every pre-snap foul and most accepted live-ball fouls.
    Returns ((ce_loss, ensemble) pre-snap, (ce_loss, ensemble) live-ball).
    """
    if raw_data is None:
        raw_data = DataManager.get_data(
            path_to_json=(Path.cwd() / "nfl/data").resolve()
        )
    data = DataManager.create_penalty_targets(
        raw_data, PRE_SNAP_PENALTIES, LIVE_BALL_PENALTIES
    )
    data = DataManager.clean_data(data, target_col="penalty")

    pre_snap_data, pre_snap_cols = DataManager.one_hot_encode_play_types(
        data, scenario_columns=SCENARIO_COLS, play_columns=PRE_SNAP_PLAY_COLS
    )
    pre_snap = _train_engine(
        PreSnapPenaltyEngine,
        pre_snap_data,
        pre_snap_cols,
        "pre_snap_penalty_target",
        "nfl/artifacts/pre_snap_penalty_ensemble.pth",
        num_epoch,
        num_splits,
        n_trials,
    )

    live_data, live_cols = DataManager.one_hot_encode_play_types(
        data[data["is_pre_snap_foul"] == 0],
        scenario_columns=SCENARIO_COLS,
        play_columns=PLAY_COLS,
    )
    live_ball = _train_engine(
        LiveBallPenaltyEngine,
        live_data,
        live_cols,
        "live_penalty_target",
        "nfl/artifacts/live_ball_penalty_ensemble.pth",
        num_epoch,
        num_splits,
        n_trials,
    )
    return pre_snap, live_ball


def _train_engine(
    model_cls: type[PenaltyEngine],
    data,
    feature_cols: list[str],
    target_col: str,
    save_path: str,
    num_epoch: int,
    num_splits: int,
    n_trials: int,
):
    print(f"\n=== {model_cls.__name__}: {len(data)} snaps ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test, groups_train = create_train_test_split(
        df=data,
        feature_cols=feature_cols,
        target_col=target_col,
        test_mask=(data["season"] == TEST_SEASON).to_numpy(),
        device=device,
        flatten_target=True,
        group_col="game_id",
    )

    # No class weights: the simulator samples these probabilities, so they need
    # to match real penalty rates
    pipeline = PipelineDriver(
        solver_cls=Solver,
        model_cls=model_cls,
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
        X=X_train, y=y_train, groups=groups_train, feature_names=feature_cols
    )
    # Computed over all rows: it only names rare penalties and is never scored
    ensemble.config["other_penalties"] = _other_penalty_frequencies(
        data, target_col, len(model_cls.PENALTIES)
    )
    ensemble.save(save_path)

    ce_loss = _report_penalty_metrics(
        model_cls, ensemble, y_train, X_test, y_test, feature_cols
    )
    return ce_loss, ensemble


def _other_penalty_frequencies(data, target_col: str, k: int) -> dict:
    """How often each penalty in the other classes occurs, by side."""
    frequencies = {}
    for side, cls_idx in (("offense", k + 1), ("defense", k + 2)):
        types = data.loc[data[target_col] == cls_idx, "penalty_type"]
        # Unmapped penalty names were filled with 0 by clean_data
        counts = types[types != 0].map(lambda p: p.name).value_counts()
        frequencies[side] = (counts / counts.sum()).to_dict()
    return frequencies


def _report_penalty_metrics(
    model_cls: type[PenaltyEngine], ensemble, y_train, X_test, y_test, feature_cols
) -> float:
    """Held-out test metrics against a baseline that predicts each class's train
    frequency for every snap (it knows the base rates but not the situation)."""
    n_classes = len(model_cls.PENALTIES) + 3
    logits, _ = ensemble(X_test)
    probs = F.softmax(logits, dim=-1)
    prior = torch.bincount(y_train, minlength=n_classes).float()
    prior = (prior / prior.sum()).expand_as(probs)

    def cross_entropy(p):
        return -p.gather(1, y_test.unsqueeze(1)).clamp_min(1e-12).log().mean().item()

    ce_loss = cross_entropy(probs)
    print(f"Test cross-entropy: {ce_loss:.5f} (base rates: {cross_entropy(prior):.5f})")
    print(
        f"P(any penalty): predicted {1 - probs[:, 0].mean().item():.4f}, "
        f"actual {(y_test != 0).float().mean().item():.4f}"
    )

    # Mean predicted rate per class should match the actual rate on test
    actual = torch.bincount(y_test, minlength=n_classes).float() / len(y_test)
    predicted = probs.mean(dim=0)
    print(f"{'class':<40}{'pred %':>8}{'act %':>8}")
    for i in actual.argsort(descending=True)[1:9].tolist():
        name = model_cls.class_names()[i]
        print(f"{name:<40}{predicted[i] * 100:>8.3f}{actual[i] * 100:>8.3f}")

    # The call should matter: pass interference is far likelier on deep throws
    if "pass_length" in feature_cols:
        dpi = model_cls.class_names().index("Defensive Pass Interference (def)")
        depth = X_test[:, feature_cols.index("pass_length")]
        for name, code in (("short", 1), ("deep", 2)):
            mask = depth == code
            pred = probs[mask, dpi].mean().item() * 100
            act = (y_test[mask] == dpi).float().mean().item() * 100
            print(f"DPI on {name} passes: predicted {pred:.2f}%, actual {act:.2f}%")
    return ce_loss
