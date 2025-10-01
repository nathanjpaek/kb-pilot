import torch
from torch import nn


class MaskUpdate(nn.Module):

    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()
        self.updateFunc = nn.ReLU(True)
        self.alpha = alpha

    def forward(self, inputMaskMap):
        return torch.pow(self.updateFunc(inputMaskMap), self.alpha)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4}]
