import torch
from torch import nn


class SimpleTwoLayer(nn.Module):
    """Some Information about SimpleTwoLayer"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleTwoLayer, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z1 = self.l1(x)
        o1 = torch.sigmoid(z1)
        z2 = self.l2(o1)
        yhat = torch.sigmoid(z2)
        return yhat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}]
