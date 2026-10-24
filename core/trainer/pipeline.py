from collections.abc import Callable

import numpy as np
import optuna
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from core.trainer.ensemble import EnsembleModel

# Hyperparameters consumed by training; every other key goes to the model constructor
TRAINING_KEYS = {"batch_size", "lr", "weight_decay"}


def _suggest_params(trial: optuna.Trial, param_space: dict) -> dict:
    """Samples each (low, high, distribution) spec; non-tuple values are fixed."""
    hparams = {}
    for k, spec in param_space.items():
        if not isinstance(spec, tuple):
            hparams[k] = spec
            continue
        low, high, distribution = spec
        if distribution == "int":
            hparams[k] = trial.suggest_int(k, low, high)
        elif distribution == "float":
            hparams[k] = trial.suggest_float(k, low, high)
        elif distribution == "log":
            hparams[k] = trial.suggest_float(k, low, high, log=True)
        elif distribution == "bool":
            hparams[k] = trial.suggest_categorical(k, [True, False])
        elif distribution == "categorical":
            hparams[k] = trial.suggest_categorical(k, low)
    return hparams


def _group_split(groups: np.ndarray, **sizes) -> tuple[np.ndarray, np.ndarray]:
    """One shuffled split that keeps each group on one side."""
    splitter = GroupShuffleSplit(n_splits=1, random_state=42, **sizes)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


class PipelineDriver:
    """Optuna search with grouped K-fold CV, then a K-model ensemble.

    model_cls can be any nn.Module whose constructor takes input_size plus the
    non-training keys of the param space. Without groups, every row is its own group.
    """

    def __init__(
        self,
        solver_cls,
        model_cls,
        criterion,
        device: str | torch.device = "cpu",
        n_splits: int = 5,
        num_epochs: int = 100,
        max_delta_threshold: float = 0.15,
        scale_features: bool = True,
        patience: int | None = 10,
        prune_warmup_epochs: int = 5,
        optimizer_cls: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
    ):
        self.solver_cls = solver_cls
        self.model_cls = model_cls
        self.criterion = criterion
        self.device = torch.device(device)
        self.n_splits = n_splits
        self.num_epochs = num_epochs
        self.max_delta_threshold = max_delta_threshold
        self.scale_features = scale_features
        self.patience = patience
        self.prune_warmup_epochs = prune_warmup_epochs
        self.optimizer_cls = optimizer_cls

        self.best_params: dict | None = None
        self.best_cv_loss: float | None = None

    def _make_solver(self, hparams: dict, input_size: int, **solver_kwargs):
        """The single place models, optimizers and solvers are built."""
        model_kwargs = {k: v for k, v in hparams.items() if k not in TRAINING_KEYS}
        model = self.model_cls(input_size=input_size, **model_kwargs).to(self.device)
        optimizer = self.optimizer_cls(
            model.parameters(),
            lr=float(hparams.get("lr", 1e-3)),
            weight_decay=float(hparams.get("weight_decay", 0.0)),
        )
        return self.solver_cls(
            model=model,
            device=self.device,
            num_epochs=self.num_epochs,
            batch_size=int(hparams.get("batch_size", 128)),
            optimizer=optimizer,
            criterion=self.criterion,
            patience=self.patience,
            **solver_kwargs,
        )

    def _fit_scaler(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-feature mean/std of X, or None when scaling is off."""
        if not self.scale_features:
            return None
        X = X.float()
        mean = X.mean(dim=0)
        std = X.std(dim=0)
        std[std < 1e-8] = 1.0  # constant columns pass through unscaled
        return mean, std

    @staticmethod
    def _apply_scaler(X: torch.Tensor, scaler) -> torch.Tensor:
        if scaler is None:
            return X
        mean, std = scaler
        return (X - mean.to(X.device)) / std.to(X.device)

    def _kfold_indices(self, groups: np.ndarray):
        """Yields (train_idx, val_idx) per fold; rows sharing a group stay together."""
        kf = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return kf.split(np.zeros(len(groups)), groups=groups)

    def _evaluate_kfolds(self, X, y, hparams: dict, groups: np.ndarray, trial) -> float:
        """Mean best validation loss across folds; pruning can stop it in any fold."""
        fold_losses = []
        for fold, (train_idx, val_idx) in enumerate(self._kfold_indices(groups)):
            solver = self._make_solver(
                hparams,
                X.shape[1],
                optuna_trial=trial,
                optuna_step_offset=fold * self.num_epochs,
                optuna_warmup_epochs=self.prune_warmup_epochs,
            )
            metrics = solver.train(
                X[train_idx], y[train_idx], X[val_idx], y[val_idx], saveBest=False
            )
            fold_losses.append(metrics["best_val_loss"])
        return float(np.mean(fold_losses))

    def select_parameters(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        param_space: dict,
        n_trials: int = 30,
        holdout_ratio: float = 0.15,
        groups=None,
        search_frac: float = 1.0,
        pruner: optuna.pruners.BasePruner | None = None,
    ) -> dict:
        """Optuna search with K-fold CV, then a holdout check of the best params.

        search_frac: fraction of the non-holdout groups used for the search, to speed
            it up. The final ensemble still trains on everything via train_networks.
        """
        groups = np.arange(len(X)) if groups is None else np.asarray(groups)
        search_idx, holdout_idx = _group_split(groups, test_size=holdout_ratio)
        if search_frac < 1.0:
            sample_idx, _ = _group_split(groups[search_idx], train_size=search_frac)
            search_idx = search_idx[sample_idx]
        groups_search = groups[search_idx]

        # Fit scaling on the search split only so the holdout check stays honest
        scaler = self._fit_scaler(X[search_idx])
        X_search = self._apply_scaler(X[search_idx], scaler)
        X_holdout = self._apply_scaler(X[holdout_idx], scaler)
        y_search, y_holdout = y[search_idx], y[holdout_idx]

        def objective(trial: optuna.Trial) -> float:
            hparams = _suggest_params(trial, param_space)
            return self._evaluate_kfolds(
                X_search, y_search, hparams, groups_search, trial
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if pruner is None:
            # Warmup is handled per fold in Solver (prune_warmup_epochs)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=3)
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=pruner
        )
        study.optimize(objective, n_trials=n_trials)

        fixed = {k: v for k, v in param_space.items() if not isinstance(v, tuple)}
        self.best_params = {**fixed, **study.best_trial.params}
        self.best_cv_loss = study.best_value
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        print(f"{n_pruned}/{len(study.trials)} trials pruned")
        print(self.best_params)

        # Backstop: retrain on the search split and compare holdout loss to CV loss.
        # Both use the best epoch's validation loss, so they're comparable.
        solver = self._make_solver(self.best_params, X.shape[1])
        metrics = solver.train(X_search, y_search, X_holdout, y_holdout, saveBest=False)
        holdout_loss = metrics["best_val_loss"]
        delta = (holdout_loss - self.best_cv_loss) / self.best_cv_loss
        print(f"CV loss {self.best_cv_loss:.6f} | holdout loss {holdout_loss:.6f}")

        if delta > self.max_delta_threshold:
            raise RuntimeError(
                f"Execution Halted: Validation delta ({delta:+.2%}) exceeded threshold "
                f"({self.max_delta_threshold:.2%}). Search space produced overfitted parameters."
            )
        return self.best_params

    def train_networks(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        save_path: str | None = None,
        groups=None,
    ) -> EnsembleModel:
        if self.best_params is None:
            raise ValueError(
                "Must call pipeline.select_parameters() before training networks."
            )

        groups = np.arange(len(X)) if groups is None else np.asarray(groups)
        scaler = self._fit_scaler(X)
        X_scaled = self._apply_scaler(X, scaler)

        models = []
        for train_idx, val_idx in self._kfold_indices(groups):
            solver = self._make_solver(self.best_params, X.shape[1])
            solver.train(
                X_scaled[train_idx],
                y[train_idx],
                X_scaled[val_idx],
                y[val_idx],
                saveBest=True,
            )
            models.append(solver.bestModel or solver.model)

        feature_mean, feature_std = scaler if scaler is not None else (None, None)
        ensemble = EnsembleModel(
            models,
            config={"input_size": X.shape[1], **self.best_params},
            feature_mean=feature_mean,
            feature_std=feature_std,
        ).to(self.device)
        if save_path:
            ensemble.save(save_path)
        return ensemble
