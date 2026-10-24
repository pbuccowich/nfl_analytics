import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feed-forward net: [Linear -> LeakyReLU -> Dropout] x N -> Linear (-> activation).

    Domain models subclass this and fix output_size / output_activation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
        relu_slope: float = 0.01,
        output_activation: nn.Module | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=relu_slope))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_size))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
