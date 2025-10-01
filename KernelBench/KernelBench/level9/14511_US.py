import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.utils import data as data
import torch.onnx


class US(nn.Module):
    """Up-sampling block
    """

    def __init__(self, num_feat, scale):
        super(US, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 1)
        self.pa_conv = nn.Conv2d(num_feat, num_feat, 1)
        self.pa_sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x_ = self.conv1(F.interpolate(x, scale_factor=self.scale, mode=
            'nearest'))
        x_ = self.lrelu(x_)
        z = self.pa_conv(x_)
        z = self.pa_sigmoid(z)
        z = torch.mul(x_, z) + x_
        z = self.conv2(z)
        out = self.lrelu(z)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_feat': 4, 'scale': 1.0}]
