import torch
import torch.nn.functional as F

from core.models.mlp import MLP
from nfl.lib.constants import PASS_YARDS_RANGE
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
    """Returns raw logits over whole-yard bins of PASS_YARDS_RANGE.

    Predicting a distribution instead of a single number lets the simulator sample
    realistic yardage, including the long tail and touchdowns. Train with
    nn.CrossEntropyLoss on bin indices (yards - low).
    """

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
            output_size=PASS_YARDS_RANGE[1] - PASS_YARDS_RANGE[0] + 1,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            relu_slope=relu_slope,
        )

    # Static so they also work on an EnsembleModel's mean logits
    @staticmethod
    def yards_proba(logits: torch.Tensor, yardline_100: torch.Tensor) -> torch.Tensor:
        """Softmax over yard bins with impossible gains masked out.

        A gain can't go past the goal line (gaining exactly yardline_100 is a
        touchdown) and a loss can't go past our own end line (a safety).
        """
        low, high = PASS_YARDS_RANGE
        yards = torch.arange(low, high + 1, device=logits.device)
        yardline = yardline_100.reshape(-1, 1).to(logits.device)
        possible = (yards <= yardline) & (yards >= yardline - 100)
        return F.softmax(logits.masked_fill(~possible, float("-inf")), dim=-1)

    @staticmethod
    def sample_yards(
        logits: torch.Tensor, yardline_100: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        """Samples yards gained (not bin indices), shape (batch, num_samples)."""
        probs = PassDistanceEngine.yards_proba(logits, yardline_100)
        return torch.multinomial(probs, num_samples=num_samples) + PASS_YARDS_RANGE[0]
