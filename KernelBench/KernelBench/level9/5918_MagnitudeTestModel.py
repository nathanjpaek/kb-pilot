import torch
from torch import nn
from torchvision import models as models
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.transforms import *
import torch.onnx


def fill_bias(module, value):
    module.bias.data.fill_(value)


def fill_conv_weight(conv, value):
    conv.weight.data.fill_(value)
    with torch.no_grad():
        mask = torch.eye(conv.kernel_size[0])
        conv.weight += mask


def create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init
    ):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    fill_conv_weight(conv, weight_init)
    fill_bias(conv, bias_init)
    return conv


class conv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(conv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=(5, 5, 3), stride=(stride, stride, 1),
            padding=(2, 2, 1), bias=False, groups=groups)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        return out


class MagnitudeTestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 2, 2, 9, -2)
        self.conv2 = create_conv(2, 1, 3, -10, 0)

    def forward(self, x):
        return self.conv2(self.conv1(x))


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
