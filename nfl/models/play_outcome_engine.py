import torch.nn as nn
import torch.nn.functional as F
from nfl.lib.nfl_enums import PassOutcome

class PassOutcomeEngine(nn.Module):
    # Given the playcall is a pass, what is the outcome?
    def __init__(self, input_size: int, hidden_size: int, num_hidden_layers: int, dropout_rate: float, relu_slope: float, **kwargs):
        super().__init__()  
        layers = []
        
        in_dim = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=relu_slope))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size
            
        # Output Layer
        layers.append(nn.Linear(hidden_size, len(PassOutcome)))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

class PassDistanceEngine(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_hidden_layers: int, dropout_rate: float, relu_slope: float, **kwargs):
        super().__init__()
        layers = []
        
        in_dim = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=relu_slope))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size
            
        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
