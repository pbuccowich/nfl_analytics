from pathlib import Path

import torch

from core.trainer.losses import CensoredCrossEntropyLoss
from core.trainer.pipeline import PipelineDriver
from core.trainer.solver import Solver
from core.utils.lib.calibration import crps
from core.utils.lib.data_handling import create_train_test_split
from nfl.data_management.data_manager import DataManager
from nfl.lib.constants import (
    EXCLUDED_PLAY_TYPES,
    PLAY_COLS,
    SCENARIO_COLS,
    TEST_SEASON,
    YARDS_RANGE,
)
from nfl.models.play_outcome_engine import RunOutcomeEngine
from nfl.train_pass_predictor import _get_parameter_grid


def train_run_engine(
    num_epoch: int = 100,
    num_splits: int = 5,
    data=None,
    n_trials: int = 30,
):
    """Trains P(yards or lost fumble | scenario, call) for designed runs.

    data is clean_data output (loaded and cleaned if not given).
    """
    if data is None:
        data = DataManager.get_data(path_to_json=(Path.cwd() / "nfl/data").resolve())
        data = DataManager.clean_data(
            data,
            excluded_play_types=EXCLUDED_PLAY_TYPES,
            target_col="epa",
        )
    data = DataManager.select_run_plays(data)
    data = DataManager.create_run_targets(data, RunOutcomeEngine.FUMBLE_LOST)
    data, feature_cols = DataManager.one_hot_encode_play_types(
        data,
        scenario_columns=SCENARIO_COLS,
        play_columns=PLAY_COLS,
    )
    n_classes = RunOutcomeEngine.N_BINS + 1
    # The pipeline passes one target column to the loss, so pack the class and the
    # censoring flags into one integer the loss unpacks
    data["run_target"] = CensoredCrossEntropyLoss.encode(
        data["run_class"],
        data["run_yards_at_least"],
        data["run_yards_at_most"],
        n_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    X_train, y_train, X_test, y_test, groups_train = create_train_test_split(
        df=data,
        feature_cols=feature_cols,
        target_col="run_target",
        test_mask=(data["season"] == TEST_SEASON).to_numpy(),
        device=device,
        flatten_target=True,
        group_col="game_id",
    )

    pipeline = PipelineDriver(
        solver_cls=Solver,
        model_cls=RunOutcomeEngine,
        criterion=CensoredCrossEntropyLoss(
            n_classes, n_ordered=RunOutcomeEngine.N_BINS
        ),
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
        save_path="nfl/artifacts/run_outcome_ensemble.pth",
        groups=groups_train,
        feature_names=feature_cols,
    )

    ce_loss = _report_run_metrics(ensemble, y_train, X_test, y_test, feature_cols)
    return ce_loss, ensemble


def _report_run_metrics(ensemble, y_train, X_test, y_test, feature_cols) -> float:
    """Held-out test metrics, against a baseline that predicts the train split's
    run histogram for every play (so it knows nothing about the situation). Both get
    the same field ends, and CE is the censored training loss. CRPS and mean yards
    are over the plays without a lost fumble, using the model's yards distribution
    given no fumble."""
    n_bins, n_classes = RunOutcomeEngine.N_BINS, RunOutcomeEngine.N_BINS + 1
    fumble = RunOutcomeEngine.FUMBLE_LOST
    criterion = CensoredCrossEntropyLoss(n_classes, n_ordered=n_bins)
    test_classes = y_test % n_classes
    actual_td = y_test // n_classes == CensoredCrossEntropyLoss.AT_LEAST
    low, high = YARDS_RANGE
    yards = torch.arange(low, high + 1, device=X_test.device)
    yardline = X_test[:, feature_cols.index("yardline_100")]
    ydstogo = X_test[:, feature_cols.index("ydstogo")]
    down = X_test[:, feature_cols.index("down")]

    logits, _ = ensemble(X_test)
    model_probs = RunOutcomeEngine.outcome_proba(logits, yardline)

    # Half-count smoothing so classes never seen in train don't get zero probability
    hist = torch.bincount(y_train % n_classes, minlength=n_classes) + 0.5
    base_logits = hist.float().log().expand(len(y_test), -1)
    base_probs = RunOutcomeEngine.outcome_proba(base_logits, yardline)

    def scores(logits, probs, mask):
        ce = criterion(logits[mask], y_test[mask]).item()
        p_fumble = probs[mask, fumble].mean().item()
        p_td = probs[mask, yardline[mask].long() - low].mean().item()
        kept = mask & (test_classes != fumble)
        p_yards = probs[kept, :n_bins]
        p_yards = p_yards / p_yards.sum(dim=1, keepdim=True)
        crps_mean = crps(p_yards, test_classes[kept]).mean().item()
        mean_yards = (p_yards * yards).sum(dim=1).mean().item()
        return ce, crps_mean, mean_yards, p_td, p_fumble

    print(
        f"{'plays':<14}{'n':>7}  {'CE model/base':>15}  {'CRPS model/base':>17}"
        f"  {'mean yds pred/act':>18}  {'P(TD) pred/act':>16}"
        f"  {'P(fumble) pred/act':>18}"
    )
    everything = torch.ones_like(y_test, dtype=torch.bool)
    groups = [
        ("all", everything),
        ("inside 10", yardline <= 10),
        ("3rd/4th & <=2", (down >= 3) & (ydstogo <= 2)),
        ("1st & 10", (down == 1) & (ydstogo == 10)),
    ]
    for name, mask in groups:
        ce_m, crps_m, yds_m, td_m, fum_m = scores(logits, model_probs, mask)
        ce_b, crps_b, _, _, _ = scores(base_logits, base_probs, mask)
        kept = mask & (test_classes != fumble)
        yds_act = yards[test_classes[kept]].float().mean().item()
        td_act = actual_td[mask].float().mean().item()
        fum_act = (test_classes[mask] == fumble).float().mean().item()
        print(
            f"{name:<14}{int(mask.sum()):>7}  {ce_m:>7.3f}/{ce_b:<7.3f}  "
            f"{crps_m:>8.3f}/{crps_b:<8.3f}  {yds_m:>8.2f}/{yds_act:<9.2f}"
            f"  {td_m:>7.4f}/{td_act:<8.4f}  {fum_m:>8.4f}/{fum_act:<9.4f}"
        )
    return scores(logits, model_probs, everything)[0]
