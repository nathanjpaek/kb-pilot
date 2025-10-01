import torch
import torch.nn as nn


class LinearRegression(nn.Module):

    def __init__(self, hidden_size):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 3)

    def forward(self, x, mask):
        y = self.linear1(x)
        y = y * mask
        return y.view(-1, 3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 3])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
