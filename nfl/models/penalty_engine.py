from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from core.models.mlp import MLP
from nfl.lib.constants import LIVE_BALL_PENALTIES, PRE_SNAP_PENALTIES
from nfl.lib.nfl_enums import PenaltyType


@dataclass(frozen=True)
class Flag:
    """A penalty called on a play."""

    penalty: PenaltyType
    on_offense: bool


class PenaltyEngine(MLP):
    """Returns raw logits over [no penalty, *PENALTIES, other offensive, other
    defensive] (use with nn.CrossEntropyLoss).

    The data records at most one penalty per play, so this is one choice of which
    penalty, if any, rather than an independent yes/no per penalty type.
    """

    PENALTIES: ClassVar[list[tuple[PenaltyType, bool]]] = []

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
            output_size=len(self.PENALTIES) + 3,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            relu_slope=relu_slope,
        )

    @classmethod
    def class_names(cls) -> list[str]:
        side = {True: "off", False: "def"}
        return [
            "none",
            *(f"{p.value.name} ({side[off]})" for p, off in cls.PENALTIES),
            "other (off)",
            "other (def)",
        ]

    # Classmethod so it also works on an EnsembleModel's mean logits
    @classmethod
    def sample_flags(
        cls,
        logits: torch.Tensor,
        other_penalties: dict[str, dict[str, float]],
        rng: np.random.Generator | None = None,
    ) -> list[Flag | None]:
        """One draw per row: None for no penalty, otherwise the Flag.

        other_penalties is the ensemble's config entry of the same name. It holds
        how often each rare penalty occurs, to name one drawn from an other class.
        """
        rng = rng or np.random.default_rng()
        drawn = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1).tolist()
        k = len(cls.PENALTIES)
        flags: list[Flag | None] = []
        for i in drawn:
            if i == 0:
                flags.append(None)
            elif i <= k:
                flags.append(Flag(*cls.PENALTIES[i - 1]))
            else:
                side = "offense" if i == k + 1 else "defense"
                names = list(other_penalties[side])
                name = rng.choice(names, p=list(other_penalties[side].values()))
                flags.append(Flag(PenaltyType[str(name)], side == "offense"))
        return flags


class PreSnapPenaltyEngine(PenaltyEngine):
    """Fouls before the snap. Inputs are the scenario and formation only, since
    no play runs and the call is never recorded."""

    PENALTIES = PRE_SNAP_PENALTIES


class LiveBallPenaltyEngine(PenaltyEngine):
    """Fouls during a snapped pass or run, given the scenario and the call."""

    PENALTIES = LIVE_BALL_PENALTIES
