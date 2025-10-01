import torch
import torch.nn as nn


class LogVarLayer(nn.Module):
    """
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    """

    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 
            1e-06, 1000000.0))


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
