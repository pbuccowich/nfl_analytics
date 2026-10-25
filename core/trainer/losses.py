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

    The first n_ordered classes are the ordered bins; any classes after them are
    plain labels (like "fumble lost") that a bound never covers. Targets come from
    encode(): class + n_classes * kind.
    """

    # Solver uses this to treat targets as integer class labels
    is_classification = True
    EXACT, AT_LEAST, AT_MOST = 0, 1, 2

    def __init__(self, n_classes: int, n_ordered: int | None = None):
        super().__init__()
        self.n_classes = n_classes
        self.n_ordered = n_classes if n_ordered is None else n_ordered

    @staticmethod
    def encode(classes, at_least, at_most, n_classes: int):
        """Works on numpy arrays, pandas Series or tensors of the same length."""
        kind = at_least * CensoredCrossEntropyLoss.AT_LEAST
        kind = kind + at_most * CensoredCrossEntropyLoss.AT_MOST
        return classes + n_classes * kind

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        classes = (target % self.n_classes).unsqueeze(1)
        kind = (target // self.n_classes).unsqueeze(1)
        idx = torch.arange(self.n_classes, device=logits.device)
        keep = torch.where(
            kind == self.AT_LEAST,
            (idx >= classes) & (idx < self.n_ordered),
            torch.where(kind == self.AT_MOST, idx <= classes, idx == classes),
        )
        log_p = F.log_softmax(logits, dim=-1).masked_fill(~keep, float("-inf"))
        return -torch.logsumexp(log_p, dim=-1).mean()
