import numpy as np
import torch
from sklearn.model_selection import KFold


class KFoldCrossValidator:
    def __init__(self, solver_cls, model_cls, criterion, device="cpu", n_splits=5, num_epochs=100):
        self.solver_cls = solver_cls
        self.model_cls = model_cls
        self.criterion = criterion
        self.device = device
        self.n_splits = n_splits
        self.num_epochs = num_epochs

    def evaluate(self, X, y, hparams, trial=None) -> float:
        """Runs K-Fold CV for a single trial and returns mean validation loss."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
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
                optuna_trial=None,  # Handled at fold level below
            )

            metrics = solver.train(X_tr, y_tr, X_val, y_val, saveBest=False)
            fold_loss = metrics["best_val_loss"]
            fold_losses.append(fold_loss)

            # Intermediate pruning: report progress to Optuna fold-by-fold
            if trial is not None:
                import optuna
                trial.report(np.mean(fold_losses), step=fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_losses))