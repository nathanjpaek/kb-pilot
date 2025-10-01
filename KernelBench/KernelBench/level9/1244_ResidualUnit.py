import torch
import torch.nn as nn


def defaultConv(inChannels, outChannels, kernelSize, bias=True):
    return nn.Conv2d(inChannels, outChannels, kernelSize, padding=
        kernelSize // 2, bias=bias)


class ResidualUnit(nn.Module):

    def __init__(self, inChannel, outChannel, reScale, kernelSize=1, bias=True
        ):
        super().__init__()
        self.reduction = defaultConv(inChannel, outChannel // 2, kernelSize,
            bias)
        self.expansion = defaultConv(outChannel // 2, inChannel, kernelSize,
            bias)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        res = self.reduction(x)
        res = self.lamRes * self.expansion(res)
        x = self.lamX * x + res
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inChannel': 4, 'outChannel': 4, 'reScale': [4, 4]}]
