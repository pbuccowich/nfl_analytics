import copy

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Solver:
    """Trains one model. Any nn.Module mapping a (batch, features) tensor to
    predictions works; classification vs regression is inferred from the criterion.
    """

    def __init__(
        self,
        model: nn.Module,
        device,
        num_epochs: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        patience: int | None = None,
        optuna_trial: optuna.Trial | None = None,
        optuna_step_offset: int = 0,
        optuna_warmup_epochs: int = 0,
    ):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.criterion = criterion
        # Stop once validation loss hasn't improved for this many epochs (None = never)
        self.patience = patience
        self.optuna_trial = optuna_trial
        # Pruning steps are offset per fold so a trial reports one continuous sequence
        self.optuna_step_offset = optuna_step_offset
        self.optuna_warmup_epochs = optuna_warmup_epochs
        self.bestModel: nn.Module | None = None
        self.is_classification = isinstance(
            criterion, (nn.CrossEntropyLoss, nn.NLLLoss)
        )

    def _dataset(self, x, y) -> TensorDataset:
        """Classification wants (N,) long targets; regression wants (N, 1) floats."""
        x = torch.as_tensor(x, dtype=torch.float32)
        if self.is_classification:
            y = torch.as_tensor(y).reshape(-1).long()
        else:
            y = torch.as_tensor(y).reshape(-1, 1).float()
        return TensorDataset(x, y)

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        """One pass over loader; returns the mean per-sample loss."""
        self.model.train(train)
        total = 0.0
        with torch.set_grad_enabled(train):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss = self.criterion(self.model(inputs), targets)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total += loss.item() * inputs.size(0)
        return total / len(loader.dataset)  # type: ignore[arg-type]

    def train(self, x, y, x_v, y_v, saveBest: bool = True) -> dict:
        train_loader = DataLoader(
            self._dataset(x, y), batch_size=self.batch_size, shuffle=True
        )
        valid_loader = DataLoader(self._dataset(x_v, y_v), batch_size=self.batch_size)

        best_valid_loss = float("inf")
        best_epoch = 0
        for epoch in range(self.num_epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            valid_loss = self._run_epoch(valid_loader, train=False)

            if self.optuna_trial is not None:
                self.optuna_trial.report(valid_loss, self.optuna_step_offset + epoch)
                if (
                    epoch >= self.optuna_warmup_epochs
                    and self.optuna_trial.should_prune()
                ):
                    raise optuna.exceptions.TrialPruned()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                if saveBest:
                    self.bestModel = copy.deepcopy(self.model)

            if epoch % 20 == 0 or epoch == self.num_epochs - 1:
                print(
                    f"Epoch {epoch:>3d}/{self.num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}"
                )

            if self.patience is not None and epoch - best_epoch >= self.patience:
                print(
                    f"Early stop at epoch {epoch} (best {best_epoch}) | "
                    f"Valid Loss: {best_valid_loss:.4f}"
                )
                break

        return {
            "best_val_loss": best_valid_loss,
            "final_train_loss": train_loss,
            "best_epoch": best_epoch,
            "epochs_run": epoch + 1,
        }
