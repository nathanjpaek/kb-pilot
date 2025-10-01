import torch
import torch.nn as nn
import torch.nn.functional as F


class FC1(nn.Module):
    """ Neural network definition
    """

    def __init__(self, size, hidden_layers):
        super(FC1, self).__init__()
        self.size = size
        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(in_features=self.size ** 2, out_features=self.
            hidden_layers)
        self.fc2 = nn.Linear(in_features=self.hidden_layers, out_features=2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4, 'hidden_layers': 1}]
