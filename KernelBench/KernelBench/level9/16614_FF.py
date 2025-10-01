import torch
from torch import nn


class FF(nn.Module):
    """
    Feed-forward in a transformer layer.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lin_1 = nn.Linear(input_size, hidden_size)
        self.lin_2 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.lin_2(self.relu(self.lin_1(x)))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
