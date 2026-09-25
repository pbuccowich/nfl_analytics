from pathlib import Path
import torch
import torch.nn as nn


class EnsembleModel(nn.Module):

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs forward passes across all K ensemble models.

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

    def save(self, filepath: str | Path) -> None:
        """Saves the ensemble state dict to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        base_models: list[nn.Module],
        map_location: str | torch.device | None = None,
    ) -> "EnsembleModel":
        """Reconstructs the ensemble from disk given initialized base model

        architectures.
        """
        ensemble = cls(base_models)
        state_dict = torch.load(
            filepath, map_location=map_location, weights_only=True
        )
        ensemble.load_state_dict(state_dict)
        return ensemble