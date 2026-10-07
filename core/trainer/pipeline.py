import inspect
import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter

from core.trainer.ensemble import EnsembleModel


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
    ):
        self.solver_cls = solver_cls
        self.model_cls = model_cls
        self.criterion = criterion
        self.device = torch.device(device) if isinstance(device, str) else device
        self.n_splits = n_splits
        self.num_epochs = num_epochs
        self.max_delta_threshold = max_delta_threshold
        self.log_dir = log_dir

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

    def _evaluate_kfolds(self, X, y, hparams: dict, trial=None) -> float:
        """Evaluates cross-validation loss across folds for hyperparameter optimization."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
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

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            # 1. Slice Fold Data (Slicing preserves underlying tensor/array shape)
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # 2. Instantiate Model and Optimizer for Current Fold
            model = self.model_cls(in_dim=in_dim, **model_kwargs).to(self.device)
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
                optuna_trial=trial if fold == 0 else None,  # Prune early on fold 0
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
    ) -> dict:
        X_search, X_holdout, y_search, y_holdout = train_test_split(
            X, y, test_size=holdout_ratio, random_state=42
        )
        X_holdout_dev, y_holdout_dev = X_holdout.to(self.device), y_holdout.to(self.device)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            hparams = {}
            for k, spec in param_space.items():
                low, high, distribution = spec[0], spec[1], spec[2]
                
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

            return self._evaluate_kfolds(X_search, y_search, hparams, trial=trial)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_trial.params
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
        )

        solver.train(X_search, y_search, X_holdout_dev, y_holdout_dev, saveBest=True)

        temp_model.eval()
        with torch.no_grad():
            preds = temp_model(X_holdout_dev)
            holdout_loss = self.criterion(preds, y_holdout_dev.view(-1, 1)).item()

        delta = (holdout_loss - self.best_cv_loss) / self.best_cv_loss

        if delta > self.max_delta_threshold:
            raise RuntimeError(
                f"Execution Halted: Validation delta ({delta:+.2%}) exceeded threshold "
                f"({self.max_delta_threshold:.2%}). Search space produced overfitted parameters."
            )

        return self.best_params

    def train_networks(self, X: torch.Tensor, y: torch.Tensor, save_path: str | None = None) -> EnsembleModel:
        if self.best_params is None:
            raise ValueError("Must call pipeline.select_parameters() before training networks.")

        models = []
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X_np, y_np = X.cpu().numpy(), y.cpu().numpy()

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_np, y_np)):
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
        ensemble = EnsembleModel(models, config=best_config)
        if save_path:
            ensemble.save(save_path)

        return ensemble