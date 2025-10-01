import torch
import torch.nn as nn


class upsample(nn.Module):

    def __init__(self):
        super(upsample, self).__init__()
        self.upsample = torch.nn.UpsamplingBilinear2d([256, 256])

    def forward(self, input):
        return (self.upsample(input) + 1.0) / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
