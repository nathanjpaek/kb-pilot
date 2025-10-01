import torch
import torch.nn as nn


class Block(nn.Module):
    """
    A ResNet module.
    """

    def __init__(self, iDim, hDim):
        super().__init__()
        self.W0 = nn.Linear(iDim, hDim)
        self.W1 = nn.Linear(hDim, iDim)

        def LS(w):
            return w.weight.numel() + w.bias.numel()
        self.parameterCount = LS(self.W0) + LS(self.W1)

    def forward(self, x):
        return (self.W1(self.W0(x).clamp(min=0)) + x).clamp(min=0.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'iDim': 4, 'hDim': 4}]
