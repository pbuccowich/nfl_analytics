import torch
import torch.nn as nn
import torch.nn.functional as F


class CensoredCrossEntropyLoss(nn.Module):
    """Cross-entropy over ordered bins where some targets are only bounds.

    A censored target says "this bin or higher" (or lower): the true value was cut
    off, like a run that would have gone farther but ended in the end zone. Plain
    cross-entropy would treat the bound as the exact answer and push the model to
    spike on it. Here a censored row's loss is -log of the total probability on its
    side of the bound, so the model is free to put that probability anywhere there.

    Targets come from encode(): bin + n_bins * kind.
    """

    # Solver uses this to treat targets as integer class labels
    is_classification = True
    EXACT, AT_LEAST, AT_MOST = 0, 1, 2

    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = n_bins

    @staticmethod
    def encode(bins, at_least, at_most, n_bins: int):
        """Works on numpy arrays, pandas Series or tensors of the same length."""
        kind = at_least * CensoredCrossEntropyLoss.AT_LEAST
        kind = kind + at_most * CensoredCrossEntropyLoss.AT_MOST
        return bins + n_bins * kind

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bins = (target % self.n_bins).unsqueeze(1)
        kind = (target // self.n_bins).unsqueeze(1)
        idx = torch.arange(self.n_bins, device=logits.device)
        keep = torch.where(
            kind == self.AT_LEAST,
            idx >= bins,
            torch.where(kind == self.AT_MOST, idx <= bins, idx == bins),
        )
        log_p = F.log_softmax(logits, dim=-1).masked_fill(~keep, float("-inf"))
        return -torch.logsumexp(log_p, dim=-1).mean()
