from pathlib import Path
import torch
import torch.nn as nn
import inspect

class EnsembleModel(nn.Module):

    def __init__(
        self, models: list[nn.Module], config: dict | None = None
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.config = config or {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            preds = torch.stack([model(x) for model in self.models], dim=0)
            mean_pred = torch.mean(preds, dim=0)
            std_pred = torch.std(preds, dim=0)
        return mean_pred, std_pred

    def save(self, filepath: str | Path) -> None:
        """Saves model weights along with initialization config and model

        count.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config,
            "num_models": len(self.models),
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        model_cls: type[nn.Module],
        map_location: str | torch.device | None = None,
    ) -> "EnsembleModel":
        checkpoint = torch.load(
            filepath, map_location=map_location, weights_only=False
        )

        config = checkpoint.get("config", {})
        num_models = checkpoint.get("num_models", len(checkpoint["state_dict"]))

        # Filter config keys so only valid WPAPredictor args are passed
        valid_args = inspect.signature(model_cls.__init__).parameters
        model_kwargs = {k: v for k, v in config.items() if k in valid_args}

        # Reconstruct base models safely
        base_models = [model_cls(**model_kwargs) for _ in range(num_models)]

        ensemble = cls(base_models, config=config)
        ensemble.load_state_dict(checkpoint["state_dict"])
        return ensemble