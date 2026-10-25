import torch
import torch.nn.functional as F

from core.models.mlp import MLP
from nfl.lib.constants import YARDS_RANGE
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
        return torch.multinomial(probs, num_samples=num_samples, replacement=True)


class PassDistanceEngine(MLP):
    """Returns raw logits over whole-yard bins of YARDS_RANGE.

    Predicting a distribution instead of a single number lets the simulator sample
    realistic yardage, including the long tail and touchdowns. Train with
    CensoredCrossEntropyLoss: touchdowns and safeties are bounds, not exact yards.
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
            output_size=YARDS_RANGE[1] - YARDS_RANGE[0] + 1,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            relu_slope=relu_slope,
        )

    # Static so they also work on an EnsembleModel's mean logits
    @staticmethod
    def yards_proba(logits: torch.Tensor, yardline_100: torch.Tensor) -> torch.Tensor:
        """Softmax over yard bins with the field's ends applied (fold_field_ends)."""
        return fold_field_ends(F.softmax(logits, dim=-1), yardline_100)

    @staticmethod
    def sample_yards(
        logits: torch.Tensor, yardline_100: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        """Samples yards gained (not bin indices), shape (batch, num_samples)."""
        probs = PassDistanceEngine.yards_proba(logits, yardline_100)
        return (
            torch.multinomial(probs, num_samples=num_samples, replacement=True)
            + YARDS_RANGE[0]
        )


class RunOutcomeEngine(MLP):
    """Returns raw logits over the whole-yard bins of YARDS_RANGE, then one last
    class for a lost fumble.

    A run has no outcome step like a pass (complete, sack, ...): the ball carrier is
    down somewhere or the ball is lost, so one softmax covers both. Train with
    CensoredCrossEntropyLoss(n_classes, n_ordered=n_bins): touchdowns and safeties
    are bounds, and the fumble class is never inside one.
    """

    N_BINS = YARDS_RANGE[1] - YARDS_RANGE[0] + 1
    FUMBLE_LOST = N_BINS  # class index

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
            output_size=self.N_BINS + 1,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            relu_slope=relu_slope,
        )

    @staticmethod
    def outcome_proba(logits: torch.Tensor, yardline_100: torch.Tensor) -> torch.Tensor:
        """Softmax over [yard bins..., fumble lost] with the field's ends applied."""
        return fold_field_ends(F.softmax(logits, dim=-1), yardline_100)

    @staticmethod
    def sample_run(
        logits: torch.Tensor, yardline_100: torch.Tensor, num_samples: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples (yards gained, fumble lost), each shape (batch, num_samples).

        Yards are 0 on a lost fumble: the median lost fumble is recovered at the
        line of scrimmage, so the simulator can turn the ball over there.
        """
        probs = RunOutcomeEngine.outcome_proba(logits, yardline_100)
        idx = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        fumble_lost = idx == RunOutcomeEngine.FUMBLE_LOST
        yards = (idx + YARDS_RANGE[0]).masked_fill(fumble_lost, 0)
        return yards, fumble_lost


def fold_field_ends(probs: torch.Tensor, yardline_100: torch.Tensor) -> torch.Tensor:
    """Moves probability on gains past the goal line onto the goal line bin (a
    touchdown: yards == yardline_100), and on losses past our own end line onto that
    bin (a safety).

    It's kept, not dropped: a 30-yard catch from the 20 is a touchdown, not an
    impossible play. probs' first columns are the YARDS_RANGE bins; any columns after
    them (like a fumble class) are left alone.
    """
    low, high = YARDS_RANGE
    yards = torch.arange(low, high + 1, device=probs.device)
    yards = F.pad(yards, (0, probs.shape[1] - len(yards)), value=low)
    is_bin = torch.arange(probs.shape[1], device=probs.device) <= high - low
    yardline = yardline_100.reshape(-1, 1).to(probs.device)
    for past, edge in (
        ((yards > yardline) & is_bin, yardline),
        ((yards < yardline - 100) & is_bin, yardline - 100),
    ):
        mass = (probs * past).sum(dim=1, keepdim=True)
        edge_bin = (edge - low).long().clamp(0, high - low)
        probs = probs.masked_fill(past, 0.0).scatter_add(1, edge_bin, mass)
    return probs
