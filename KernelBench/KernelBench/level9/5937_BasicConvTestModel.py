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


class BasicConvTestModel(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, kernel_size=2,
        weight_init=-1, bias_init=-2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.conv = create_conv(in_channels, out_channels, kernel_size,
            weight_init, bias_init)

    @staticmethod
    def default_weight():
        return torch.tensor([[[[0.0, -1.0], [-1.0, 0.0]]], [[[0.0, -1.0], [
            -1.0, 0.0]]]])

    @staticmethod
    def default_bias():
        return torch.tensor([-2.0, -2])

    def forward(self, x):
        return self.conv(x)

    @property
    def weights_num(self):
        return self.out_channels * self.kernel_size ** 2

    @property
    def bias_num(self):
        return self.kernel_size

    @property
    def nz_weights_num(self):
        return self.kernel_size * self.out_channels

    @property
    def nz_bias_num(self):
        return self.kernel_size


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
