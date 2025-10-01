import torch
import torch.nn as nn


class Gate(nn.Module):

    def __init__(self, input_dim):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_dim * 4, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        z = torch.cat([x, y, x * y, x - y], dim=2)
        return self.sigmoid(self.linear(z))


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
