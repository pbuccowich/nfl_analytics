import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """x + f(x), with f = LayerNorm -> Linear -> LeakyReLU -> Dropout -> Linear.

    The skip path (x) carries the input and its gradient through unchanged, so
    stacking many blocks doesn't weaken the gradient the way a plain stack does.
    The block only has to learn a correction f(x) to what it receives.

    Normalizing inside f ("pre-norm") instead of after the sum keeps the skip path
    untouched, which is what lets very deep stacks train (used in modern
    transformers too).
    """

    def __init__(self, width: int, dropout_rate: float, relu_slope: float):
        super().__init__()
        self.f = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.LeakyReLU(negative_slope=relu_slope),
            nn.Dropout(dropout_rate),
            nn.Linear(width, width),
        )
        # Start every block as the identity (f(x) = 0): a fresh deep net then
        # behaves like a shallow one and each block grows in only if it helps
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


class ResidualMLP(nn.Module):
    """Linear (input -> hidden_size) -> ResidualBlock x N -> LayerNorm -> Linear.

    The first Linear maps the inputs to the block width, since x + f(x) needs both
    to be the same size. Each block has two Linear layers, so N blocks have about
    as many weights as a plain MLP with 2N hidden layers of the same width.
    Domain models subclass this and fix output_size / output_activation, like MLP.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        relu_slope: float = 0.01,
        output_activation: nn.Module | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_size, hidden_size)]
        layers += [
            ResidualBlock(hidden_size, dropout_rate, relu_slope)
            for _ in range(num_blocks)
        ]
        # The sum of many blocks can drift in scale; normalize once before the head
        head = nn.Linear(hidden_size, output_size)
        # LayerNorm outputs are unit scale, so a default-initialized head starts
        # with outputs around +-0.5, far off for a small target like WPA. Starting
        # at 0 (uniform logits for a classifier) avoids wasting early epochs, which
        # the search's pruner judges trials on.
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers += [nn.LayerNorm(hidden_size), head]
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
