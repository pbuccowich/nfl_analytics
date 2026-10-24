import torch.nn as nn

from core.models.mlp import MLP


class WPAPredictor(MLP):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
        relu_slope: float = 0.01,
        use_tanh_output: bool = False,
    ):
        # WPA lies in [-1, 1], so tanh can bound the output
        super().__init__(
            input_size,
            output_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate,
            relu_slope=relu_slope,
            output_activation=nn.Tanh() if use_tanh_output else None,
        )
