import torch
import torch.nn as nn
import torch.utils.model_zoo


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=
        kernel_size // 2, bias=bias)


class ResidualUnit(nn.Module):

    def __init__(self, inChannel, outChannel, reScale, kernelSize=1, bias=True
        ):
        super().__init__()
        self.reduction = default_conv(inChannel, outChannel // 2,
            kernelSize, bias)
        self.expansion = default_conv(outChannel // 2, inChannel,
            kernelSize, bias)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        res = self.reduction(x)
        res = self.lamRes * self.expansion(res)
        x = self.lamX * x + res
        return x


class ARFB(nn.Module):

    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()
        self.RU1 = ResidualUnit(inChannel, outChannel, reScale)
        self.RU2 = ResidualUnit(inChannel, outChannel, reScale)
        self.conv1 = default_conv(2 * inChannel, 2 * outChannel, kernel_size=1)
        self.conv3 = default_conv(2 * inChannel, outChannel, kernel_size=3)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        x_ru1 = self.RU1(x)
        x_ru2 = self.RU2(x_ru1)
        x_ru = torch.cat((x_ru1, x_ru2), 1)
        x_ru = self.conv1(x_ru)
        x_ru = self.conv3(x_ru)
        x_ru = self.lamRes * x_ru
        x = x * self.lamX + x_ru
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inChannel': 4, 'outChannel': 4, 'reScale': [4, 4]}]
