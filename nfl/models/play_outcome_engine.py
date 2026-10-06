import torch
import torch.nn as nn
import torch.nn.functional as F
from nfl.lib.nfl_enums import PassOutcome

class PassOutcomeEngine(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_hidden_layers: int, 
        dropout_rate: float, 
        relu_slope: float,
        **kwargs
    ):
        super().__init__()  
        layers = []
        
        in_dim = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=relu_slope))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size
            
        # Output Layer returns raw unnormalized logits
        layers.append(nn.Linear(hidden_size, len(PassOutcome)))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (use with nn.CrossEntropyLoss during training)."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Computes softmax probabilities across pass outcome categories."""
        logits = self.forward(x)
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)
        return F.softmax(logits, dim=-1)

    def predict_outcome(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the single most likely outcome index (deterministic)."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)

    def sample_outcome(self, x: torch.Tensor, num_samples: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """Samples outcome indices based on the predicted probability distribution."""
        probs = self.predict_proba(x, temperature=temperature)
        
        # torch.multinomial samples indices according to input probability weights
        # Shape: (batch_size, num_samples)
        sampled_indices = torch.multinomial(probs, num_samples=num_samples)
        return sampled_indices

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
