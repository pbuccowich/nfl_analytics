import torch.nn as nn

class WPAPredictor(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 128, 
        num_hidden_layers: int = 2, 
        dropout_rate: float = 0.1,
        use_tanh_output: bool = False
    ):
        super().__init__()
        layers = []
        
        in_dim = input_size
        
        # Build Hidden Layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size
            
        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))
        
        # since wpa is bounded between -1 and 1
        if use_tanh_output:
            layers.append(nn.Tanh())
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)