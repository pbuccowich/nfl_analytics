import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
import optuna


class EnsembleModel(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs forward passes across all K ensemble models.
        Returns:
            mean_pred: Averaged prediction across ensemble
            std_pred: Standard deviation (uncertainty estimate)
        """
        self.eval()
        with torch.no_grad():
            preds = torch.stack([model(x) for model in self.models], dim=0)
            mean_pred = torch.mean(preds, dim=0)
            std_pred = torch.std(preds, dim=0)

        return mean_pred, std_pred