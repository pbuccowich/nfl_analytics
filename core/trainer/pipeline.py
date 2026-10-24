import inspect
import numpy as np
import optuna
import torch
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)
from torch.utils.tensorboard import SummaryWriter

from core.trainer.ensemble import EnsembleModel


def _suggest_params(trial: optuna.Trial, param_space: dict, skip: dict) -> dict:
    """Samples one value per (low, high, distribution) spec, skipping keys in skip."""
    hparams = {}
    for k, (low, high, distribution) in param_space.items():
        if k in skip:
            continue
        elif distribution == "int":
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


class PipelineDriver:
    def __init__(
        self,
        solver_cls,
        model_cls,
        criterion,
        device: str | torch.device = "cpu",
        n_splits: int = 5,
        num_epochs: int = 100,
        max_delta_threshold: float = 0.15,
        log_dir: str | None = None,
        scale_features: bool = True,
        patience: int | None = 10,
        prune_warmup_epochs: int = 5,
    ):
        self.solver_cls = solver_cls
        self.model_cls = model_cls
        self.criterion = criterion
        self.is_classification = isinstance(criterion, torch.nn.CrossEntropyLoss)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.n_splits = n_splits
        self.num_epochs = num_epochs
        self.max_delta_threshold = max_delta_threshold
        self.log_dir = log_dir
        self.scale_features = scale_features
        self.patience = patience
        self.prune_warmup_epochs = prune_warmup_epochs

        self.best_params: dict | None = None
        self.best_cv_loss: float | None = None

    def _instantiate_model(self, input_size: int, hparams: dict) -> torch.nn.Module:
        """Dynamically filters hparams to match model_cls signature."""
        sig = inspect.signature(self.model_cls.__init__)
        accepted_args = set(sig.parameters.keys()) - {"self"}

        model_kwargs = {"input_size": input_size}
        for k, v in hparams.items():
            if k in accepted_args:
                model_kwargs[k] = type(v)(v)

        return self.model_cls(**model_kwargs).to(self.device)

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

    def _kfold_indices(self, n_samples: int, groups=None):
        """Yields (train_idx, val_idx) per fold; rows sharing a group stay together."""
        if groups is None:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        else:
            kf = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return kf.split(np.zeros(n_samples), groups=groups)

    def _evaluate_kfolds(self, X, y, hparams: dict, trial=None, groups=None) -> float:
        """Evaluates cross-validation loss across folds for hyperparameter optimization."""
        fold_losses = []

        # Format batch size and learning rate from hyperparameters
        batch_size = int(hparams.get("batch_size", 128))
        lr = float(hparams.get("lr", 1e-3))
        weight_decay = float(hparams.get("weight_decay", 0.0))

        # Convert non-model kwargs to model-specific kwargs
        model_kwargs = {
            k: v
            for k, v in hparams.items()
            if k not in {"batch_size", "lr", "weight_decay"}
        }

        # Ensure input dimension matches features
        in_dim = X.shape[1] if isinstance(X, (np.ndarray, torch.Tensor)) else len(X[0])

        for fold, (train_idx, val_idx) in enumerate(self._kfold_indices(len(X), groups)):
            # 1. Slice Fold Data (Slicing preserves underlying tensor/array shape)
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # 2. Instantiate Model and Optimizer for Current Fold
            model = self.model_cls(input_size=in_dim, **model_kwargs).to(self.device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            # 3. Instantiate Solver
            solver = self.solver_cls(
                model=model,
                device=self.device,
                num_epochs=self.num_epochs,
                batch_size=batch_size,
                optimizer=optimizer,
                criterion=self.criterion,
                patience=self.patience,
                # Prune in every fold; a bad fold ends the trial before later folds run
                optuna_trial=trial,
                optuna_step_offset=fold * self.num_epochs,
                optuna_warmup_epochs=self.prune_warmup_epochs,
            )

            # 4. Train Fold Model (saveBest=False to prevent writing fold checkpoints)
            metrics = solver.train(X_tr, y_tr, X_val, y_val, saveBest=False)
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

        search_frac: fraction of the non-holdout data (whole groups when given)
            used for the search, to speed it up. The final ensemble still trains
            on everything via train_networks.
        """
        if groups is None:
            X_search, X_holdout, y_search, y_holdout = train_test_split(
                X, y, test_size=holdout_ratio, random_state=42
            )
            groups_search = None
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=42)
            search_idx, holdout_idx = next(splitter.split(np.zeros(len(X)), groups=groups))
            X_search, X_holdout = X[search_idx], X[holdout_idx]
            y_search, y_holdout = y[search_idx], y[holdout_idx]
            groups_search = groups[search_idx]

        if search_frac < 1.0:
            if groups_search is None:
                sample_idx, _ = train_test_split(
                    np.arange(len(X_search)), train_size=search_frac, random_state=42
                )
            else:
                splitter = GroupShuffleSplit(n_splits=1, train_size=search_frac, random_state=42)
                sample_idx, _ = next(splitter.split(np.zeros(len(X_search)), groups=groups_search))
                groups_search = groups_search[sample_idx]
            X_search, y_search = X_search[sample_idx], y_search[sample_idx]

        # Fit scaling on the search split only so the holdout check stays honest
        scaler = self._fit_scaler(X_search)
        X_search = self._apply_scaler(X_search, scaler)
        X_holdout = self._apply_scaler(X_holdout, scaler)
        X_holdout_dev, y_holdout_dev = X_holdout.to(self.device), y_holdout.to(self.device)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # A bool spec with low == high is a fixed value, not a search dimension
        fixed_params = {
            k: spec[0]
            for k, spec in param_space.items()
            if spec[2] == "bool" and spec[0] == spec[1]
        }

        def objective(trial: optuna.Trial) -> float:
            hparams = {**fixed_params, **_suggest_params(trial, param_space, fixed_params)}
            return self._evaluate_kfolds(X_search, y_search, hparams, trial=trial, groups=groups_search)

        if pruner is None:
            # Warmup is handled per fold in Solver (prune_warmup_epochs)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=3)
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=pruner
        )
        study.optimize(objective, n_trials=n_trials)

        self.best_params = {**fixed_params, **study.best_trial.params}
        self.best_cv_loss = study.best_value

        # Verification step dynamically built
        temp_model = self._instantiate_model(X.shape[1], self.best_params)

        optimizer = torch.optim.Adam(
            temp_model.parameters(),
            lr=float(self.best_params["lr"]),
            weight_decay=float(self.best_params.get("weight_decay", 0.0)),
        )

        solver = self.solver_cls(
            model=temp_model,
            device=self.device,
            num_epochs=self.num_epochs,
            batch_size=int(self.best_params["batch_size"]),
            optimizer=optimizer,
            criterion=self.criterion,
            patience=self.patience,
        )

        solver.train(X_search, y_search, X_holdout_dev, y_holdout_dev, saveBest=True)

        temp_model.eval()
        with torch.no_grad():
            preds = temp_model(X_holdout_dev)
            if self.is_classification:
                target_eval = y_holdout_dev.squeeze().long()
            else:
                target_eval = y_holdout_dev.view(-1, 1).float()
                
            holdout_loss = self.criterion(preds, target_eval).item()

        delta = (holdout_loss - self.best_cv_loss) / self.best_cv_loss

        if delta > self.max_delta_threshold:
            raise RuntimeError(
                f"Execution Halted: Validation delta ({delta:+.2%}) exceeded threshold "
                f"({self.max_delta_threshold:.2%}). Search space produced overfitted parameters."
            )

        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        print(f"{n_pruned}/{len(study.trials)} trials pruned")
        print(self.best_params)

        return self.best_params

    def train_networks(
        self, X: torch.Tensor, y: torch.Tensor, save_path: str | None = None, groups=None
    ) -> EnsembleModel:
        if self.best_params is None:
            raise ValueError("Must call pipeline.select_parameters() before training networks.")

        models = []
        scaler = self._fit_scaler(X)
        X_np, y_np = self._apply_scaler(X, scaler).cpu().numpy(), y.cpu().numpy()

        for fold, (train_idx, val_idx) in enumerate(self._kfold_indices(len(X_np), groups)):
            X_tr, y_tr = torch.tensor(X_np[train_idx], device=self.device), torch.tensor(y_np[train_idx], device=self.device)
            X_val, y_val = torch.tensor(X_np[val_idx], device=self.device), torch.tensor(y_np[val_idx], device=self.device)

            model = self._instantiate_model(X.shape[1], self.best_params)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(self.best_params["lr"]),
                weight_decay=float(self.best_params.get("weight_decay", 0.0)),
            )

            writer = SummaryWriter(f"{self.log_dir}/fold_{fold}") if self.log_dir else None

            solver = self.solver_cls(
                model=model,
                device=self.device,
                num_epochs=self.num_epochs,
                batch_size=int(self.best_params["batch_size"]),
                optimizer=optimizer,
                criterion=self.criterion,
                patience=self.patience,
            )

            metrics = solver.train(X_tr, y_tr, X_val, y_val, saveBest=True)

            if writer:
                writer.add_scalar("Loss/val", metrics["best_val_loss"], fold)
                writer.close()

            models.append(solver.bestModel if solver.bestModel else model)

        best_config = {
            "input_size": X.shape[1],
            **self.best_params,
        }
        feature_mean, feature_std = scaler if scaler is not None else (None, None)
        ensemble = EnsembleModel(
            models,
            config=best_config,
            feature_mean=feature_mean,
            feature_std=feature_std,
        ).to(self.device)
        if save_path:
            ensemble.save(save_path)

        return ensemble