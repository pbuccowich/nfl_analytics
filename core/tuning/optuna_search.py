import optuna
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Union

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaSearch:
    def __init__(
        self,
        param_grid: dict,
        solver_cls,
        model_cls,
        criterion,
        device: Union[str, torch.device] = "cpu",
        num_epochs: int = 100,
        input_size: int = 10,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        log_dir: str = "runs/optuna_sweep",
        prune: bool = True,
    ):
        self.param_grid = param_grid
        self.solver_cls = solver_cls
        self.model_cls = model_cls
        self.criterion = criterion
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_epochs = num_epochs
        self.input_size = input_size
        self.betas = betas
        self.eps = eps
        self.log_dir = log_dir
        self.prune = prune

    def _sample_params(self, trial: optuna.Trial) -> dict:
        """
        Dynamically samples parameters based on input type in param_grid:
          - Categorical / Discrete lists (e.g. batch_size: [16, 32, 64]) -> suggest_categorical
          - Float range tuples (e.g. lr: (1e-5, 1e-1)) -> suggest_float (log=True if range spans > 2 orders of magnitude)
          - Int range tuples (e.g. hidden_size: (32, 256)) -> suggest_int
        """
        hparams = {}
        for key, value in self.param_grid.items():
            if isinstance(value, tuple) and len(value) == 2:
                low, high = value
                if isinstance(low, float) or isinstance(high, float):
                    # Use log scale automatically for spans across multiple orders of magnitude
                    use_log = (low > 0 and high / low >= 100)
                    hparams[key] = trial.suggest_float(key, float(low), float(high), log=use_log)
                elif isinstance(low, int) and isinstance(high, int):
                    hparams[key] = trial.suggest_int(key, low, high)
                else:
                    hparams[key] = trial.suggest_categorical(key, value)
            elif isinstance(value, list):
                hparams[key] = trial.suggest_categorical(key, value)
            else:
                raise ValueError(f"Unsupported format for param '{key}': {value}. Use a list or a (min, max) tuple.")
        
        return hparams

    def _objective(self, trial: optuna.Trial, X_tr, y_tr, X_v, y_v) -> float:
        hparams = self._sample_params(trial)

        run_name = f"trial_{trial.number}_bs{hparams['batch_size']}_lr{hparams['lr']:.1e}"
        writer = SummaryWriter(log_dir=f"{self.log_dir}/{run_name}")

        model = self.model_cls(
            input_size=self.input_size,
            num_hidden_layers=int(hparams["num_hidden_layers"]),
            hidden_size=int(hparams["hidden_size"]),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(hparams["lr"]),
            betas=self.betas,
            eps=self.eps,
            weight_decay=float(hparams["weight_decay"]),
        )

        solver = self.solver_cls(
            model=model,
            device=self.device,
            num_epochs=self.num_epochs,
            batch_size=int(hparams["batch_size"]),
            optimizer=optimizer,
            criterion=self.criterion,
            writer=writer,
            run_name=run_name,
            hparams=hparams,
            optuna_trial=trial if self.prune else None,
        )

        metrics = solver.train(X_tr, y_tr, X_v, y_v, modelName=run_name, saveBest=True)
        writer.close()

        return metrics["best_val_loss"]

    def run(self, X_tr, y_tr, X_v, y_v, n_trials: int = 50, timeout: float | None = None) -> tuple[pd.DataFrame, dict]:
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            if self.prune
            else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=pruner,
        )

        print(f"--- Starting Optuna Search ({n_trials} trials) ---")

        study.optimize(
            lambda trial: self._objective(trial, X_tr, y_tr, X_v, y_v),
            n_trials=n_trials,
            timeout=timeout,
        )

        df = study.trials_dataframe()
        best_config = study.best_trial.params
        best_config["best_val_loss"] = study.best_value

        return df, best_config