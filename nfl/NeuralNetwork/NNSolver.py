import copy
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class Solver:

    def __init__(
        self,
        model,
        device,
        num_epochs,
        batch_size,
        optimizer,
        criterion,
        writer: SummaryWriter = None,
        run_name: str = None,
        hparams: dict = None,
        optuna_trial=None,
    ):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.run_name = run_name or "default_run"
        self.hparams = hparams
        self.optuna_trial = optuna_trial
        self.bestModel = None

    def train(self, x, y, x_v, y_v, modelName=None, saveBest=True) -> dict:
        train_ds = TensorDataset(
            (
                x.detach().clone().to(torch.float32)
                if isinstance(x, torch.Tensor)
                else torch.tensor(x, dtype=torch.float32)
            ),
            (
                y.detach().clone().to(torch.float32)
                if isinstance(y, torch.Tensor)
                else torch.tensor(y, dtype=torch.float32)
            ),
        )
        valid_ds = TensorDataset(
            (
                x_v.detach().clone().to(torch.float32)
                if isinstance(x_v, torch.Tensor)
                else torch.tensor(x_v, dtype=torch.float32)
            ),
            (
                y_v.detach().clone().to(torch.float32)
                if isinstance(y_v, torch.Tensor)
                else torch.tensor(y_v, dtype=torch.float32)
            ),
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=self.batch_size, shuffle=False
        )

        training_loss = np.empty(shape=(self.num_epochs,))
        validation_loss = np.empty(shape=(self.num_epochs,))
        best_valid_loss = float("inf")
        best_epoch = 0

        for epoch in range(self.num_epochs):
            # --- Training Phase ---
            self.model.train()
            train_running_loss = 0.0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_running_loss += loss.item() * inputs.size(0)

            epoch_train_loss = train_running_loss / len(train_ds)
            training_loss[epoch] = epoch_train_loss

            # --- Validation Phase ---
            self.model.eval()
            valid_running_loss = 0.0

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device).view(-1, 1)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    valid_running_loss += loss.item() * inputs.size(0)

            epoch_valid_loss = valid_running_loss / len(valid_ds)
            validation_loss[epoch] = epoch_valid_loss

            # --- Optuna Intermediate Pruning ---
            if self.optuna_trial is not None:
                import optuna
                self.optuna_trial.report(epoch_valid_loss, epoch)
                if self.optuna_trial.should_prune():
                    raise optuna.Exceptions.TrialPruned()

            # --- Unified TensorBoard Logging ---
            if self.writer:
                self.writer.add_scalars("Loss/Train", {self.run_name: epoch_train_loss}, epoch)
                self.writer.add_scalars("Loss/Validation", {self.run_name: epoch_valid_loss}, epoch)

            # --- Model Checkpointing ---
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                best_epoch = epoch
                if saveBest:
                    self.bestModel = copy.deepcopy(self.model)

            if epoch % 20 == 0 or epoch == self.num_epochs - 1:
                print(
                    f"Epoch {epoch:>3d}/{self.num_epochs} | "
                    f"Train Loss: {epoch_train_loss:.4f} | "
                    f"Valid Loss: {epoch_valid_loss:.4f}"
                )

            if self.writer:
                self.writer.flush()

        final_train_loss = float(training_loss[-1])

        # --- TensorBoard HParams Logging ---
        if self.writer and self.hparams:
            self.writer.add_hparams(
                hparam_dict=self.hparams,
                metric_dict={
                    "hparam/best_val_loss": float(best_valid_loss),
                    "hparam/final_train_loss": final_train_loss,
                },
                run_name=self.run_name,
            )

        if modelName and self.bestModel is not None:
            save_path = pathlib.Path(__file__).parent / f"{modelName}.pth"
            torch.save(self.bestModel.state_dict(), save_path)

        return {
            "best_val_loss": float(best_valid_loss),
            "final_train_loss": final_train_loss,
            "best_epoch": int(best_epoch),
        }