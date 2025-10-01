import torch
from torch import nn
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed


class ResNetBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride, downsample, pad,
        dilation):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
            stride=1, padding=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = x + out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'stride': 1,
        'downsample': 4, 'pad': 4, 'dilation': 1}]
