import torch
import torch.nn as nn


class VariantSigmoid(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        y = 1 / (1 + torch.exp(-self.alpha * x))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4}]
