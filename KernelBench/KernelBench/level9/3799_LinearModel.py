import torch
import torch.nn.functional as F
import torch.nn as nn


class LinearModel(nn.Module):
    """Model creation.
    """

    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, output_dim)

    def forward(self, x):
        """Forward pass."""
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
