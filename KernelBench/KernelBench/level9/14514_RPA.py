import torch
from torch import nn as nn
from torch.nn import init as init
from torch.utils import data as data
import torch.onnx


class RPA(nn.Module):
    """Residual pixel-attention block
    """

    def __init__(self, num_feat):
        super(RPA, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 1)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, 1)
        self.conv3 = nn.Conv2d(num_feat * 4, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for layer in [self.conv1, self.conv2, self.conv3, self.conv3]:
            init.kaiming_normal_(layer.weight)
            layer.weight.data *= 0.1

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        z = self.conv3(z)
        z = self.sigmoid(z)
        z = x * z + x
        z = self.conv4(z)
        out = self.lrelu(z)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_feat': 4}]
