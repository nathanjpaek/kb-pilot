import torch
from torch import nn


class NormScaleFeature(nn.Module):

    def __init__(self, init_value=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        magnitudes = 1e-06 + torch.sqrt(torch.sum(input ** 2, axis=1,
            keepdims=True))
        output = self.scale * input / magnitudes
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
