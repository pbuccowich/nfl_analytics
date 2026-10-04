from torch import nn

class PassOutcomeEngine(nn.Module):
    # Given the playcall is a pass, what is the outcome?
    def __init__(self, input_size: int, hidden_size: int, num_hidden_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, 1))