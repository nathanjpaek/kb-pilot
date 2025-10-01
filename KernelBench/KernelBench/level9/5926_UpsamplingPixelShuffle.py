import torch
from torch import nn
from torchvision import models as models
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.transforms import *
import torch.onnx


class shuffle(nn.Module):

    def __init__(self, ratio):
        super(shuffle, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        batch_size, in_channels, d, h, w = x.shape
        out_channels = in_channels // (self.ratio * self.ratio * self.ratio)
        out = x.view(batch_size * out_channels, self.ratio, self.ratio,
            self.ratio, d, h, w)
        out = out.permute(0, 4, 1, 5, 2, 6, 3)
        return out.contiguous().view(batch_size, out_channels, d * self.
            ratio, h * self.ratio, w * self.ratio)


class UpsamplingPixelShuffle(nn.Module):

    def __init__(self, input_channels, output_channels, ratio=2):
        super(UpsamplingPixelShuffle, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv3d(in_channels=input_channels, out_channels=
            output_channels * int(ratio ** 3), kernel_size=1, padding=0,
            bias=False)
        self.relu = nn.LeakyReLU(0.02, inplace=True)
        self.shuffle = shuffle(ratio=ratio)

    def forward(self, x):
        out = self.conv(x)
        out = self.shuffle(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
