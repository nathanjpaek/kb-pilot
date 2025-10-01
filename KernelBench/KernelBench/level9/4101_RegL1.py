import torch
import torch.nn as nn


class RegL1(nn.Module):
    """
    Run Regression with L1
    """

    def __init__(self, n_input, n_output):
        super(RegL1, self).__init__()
        self.linear = nn.Linear(n_input, n_output, bias=True)

    def forward(self, x, training=True):
        self.training = training
        x = self.linear(x)
        z1 = torch.sum(torch.abs(self.linear.weight))
        self.training = True
        return x, z1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input': 4, 'n_output': 4}]
