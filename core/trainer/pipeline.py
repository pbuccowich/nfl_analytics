import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from core.trainer.ensemble import EnsembleModel
import optuna
from torch.utils.tensorboard import SummaryWriter


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

    def _evaluate_kfolds(self, X: torch.Tensor, y: torch.Tensor, hparams: dict, trial=None) -> float:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold_losses = []

        X_np, y_np = X.cpu().numpy(), y.cpu().numpy()

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_np, y_np)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = self.model_cls(
                input_size=X.shape[1],
                num_hidden_layers=int(hparams["num_hidden_layers"]),
                hidden_size=int(hparams["hidden_size"]),
            ).to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(hparams["lr"]),
                weight_decay=float(hparams["weight_decay"]),
            )

            solver = self.solver_cls(
                model=model,
                device=self.device,
                num_epochs=self.num_epochs,
                batch_size=int(hparams["batch_size"]),
                optimizer=optimizer,
                criterion=self.criterion,
            )

            metrics = solver.train(X_tr, y_tr, X_val, y_val, saveBest=False)
            fold_losses.append(metrics["best_val_loss"])

            if trial is not None:
                trial.report(np.mean(fold_losses), step=fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_losses))

    def select_parameters(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        param_grid: dict,
        n_trials: int = 30,
        holdout_ratio: float = 0.15,
    ) -> dict:
        X_search, X_holdout, y_search, y_holdout = train_test_split(
            X.cpu().numpy(), y.cpu().numpy(), test_size=holdout_ratio, random_state=42
        )
        X_search_t, y_search_t = torch.tensor(X_search), torch.tensor(y_search)
        X_holdout_t, y_holdout_t = torch.tensor(X_holdout).to(self.device), torch.tensor(y_holdout).to(self.device)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            hparams = {}
            for k, v in param_grid.items():
                if isinstance(v, tuple) and len(v) == 2:
                    use_log = isinstance(v[0], float) and v[0] > 0 and (v[1] / v[0] >= 100)
                    hparams[k] = (
                        trial.suggest_float(k, v[0], v[1], log=use_log)
                        if isinstance(v[0], float)
                        else trial.suggest_int(k, v[0], v[1])
                    )
                else:
                    hparams[k] = trial.suggest_categorical(k, v)

            return self._evaluate_kfolds(X_search_t, y_search_t, hparams, trial=trial)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_trial.params
        self.best_cv_loss = study.best_value

        # Verification step
        temp_model = self.model_cls(
            input_size=X.shape[1],
            num_hidden_layers=int(self.best_params["num_hidden_layers"]),
            hidden_size=int(self.best_params["hidden_size"]),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            temp_model.parameters(),
            lr=float(self.best_params["lr"]),
            weight_decay=float(self.best_params["weight_decay"]),
        )

        solver = self.solver_cls(
            model=temp_model,
            device=self.device,
            num_epochs=self.num_epochs,
            batch_size=int(self.best_params["batch_size"]),
            optimizer=optimizer,
            criterion=self.criterion,
        )

        solver.train(X_search_t, y_search_t, X_holdout_t, y_holdout_t, saveBest=True)

        temp_model.eval()
        with torch.no_grad():
            preds = temp_model(X_holdout_t)
            holdout_loss = self.criterion(preds, y_holdout_t.view(-1, 1)).item()

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
            X_tr, y_tr = torch.tensor(X_np[train_idx]), torch.tensor(y_np[train_idx])
            X_val, y_val = torch.tensor(X_np[val_idx]), torch.tensor(y_np[val_idx])

            model = self.model_cls(
                input_size=X.shape[1],
                num_hidden_layers=int(self.best_params["num_hidden_layers"]),
                hidden_size=int(self.best_params["hidden_size"]),
            ).to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(self.best_params["lr"]),
                weight_decay=float(self.best_params["weight_decay"]),
            )

            writer = SummaryWriter(f"{self.log_dir}/fold_{fold}") if self.log_dir and SummaryWriter else None

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

        ensemble = EnsembleModel(models)

        if save_path:
            torch.save(ensemble.state_dict(), save_path)

        return ensemble