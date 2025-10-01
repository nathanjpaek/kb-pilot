import torch
from torch import nn
import torch.utils.data
import torch.nn.init


class NoiseZ(nn.Module):

    def __init__(self, batchSize):
        super(NoiseZ, self).__init__()
        self.Z = nn.Parameter(torch.randn(batchSize, 128), requires_grad=True)

    def forward(self, input):
        out = self.Z * input
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 128])]


def get_init_inputs():
    return [[], {'batchSize': 4}]
