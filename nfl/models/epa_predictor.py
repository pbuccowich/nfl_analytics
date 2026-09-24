from torch import nn

class EPAPredictor(nn.Module):
    def __init__(self, input_size, hidden_size: int = 64, num_hidden_layers: int = 1):
        super(EPAPredictor, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.LeakyReLU())
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(hidden_size, 1))

    def forward(self, x):
        for layer in self.layers:
            x = nn.functional.relu(layer(x))
        return x
