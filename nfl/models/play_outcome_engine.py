import torch
import torch.nn.functional as F

from core.models.mlp import MLP
from nfl.lib.nfl_enums import PassOutcome


class PassOutcomeEngine(MLP):
    """Returns raw logits over PassOutcome (use with nn.CrossEntropyLoss)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
        relu_slope: float = 0.01,
    ):
        super().__init__(
            input_size,
            output_size=len(PassOutcome),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            relu_slope=relu_slope,
        )

    def predict_proba(self, x: torch.Tensor, temperature: float = 1.5) -> torch.Tensor:
        """Computes softmax probabilities across pass outcome categories."""
        logits = self.forward(x) / max(temperature, 1e-5)
        return F.softmax(logits, dim=-1)

    def sample_outcome(
        self, x: torch.Tensor, num_samples: int = 1, temperature: float = 1.0
    ) -> torch.Tensor:
        """Samples outcome indices based on the predicted probability distribution."""
        probs = self.predict_proba(x, temperature=temperature)
        return torch.multinomial(probs, num_samples=num_samples)


class PassDistanceEngine(MLP):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
        relu_slope: float = 0.01,
    ):
        super().__init__(
            input_size,
            output_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            relu_slope=relu_slope,
        )
