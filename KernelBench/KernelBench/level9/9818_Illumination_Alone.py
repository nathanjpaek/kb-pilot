from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k,
        stride=s, padding=p, dilation=dilation, groups=groups)


class Illumination_Alone(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=1, out_c=32, k=5, s=1, p=2)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv5 = get_conv2d_layer(in_c=32, out_c=1, k=1, s=1, p=0)
        self.leaky_relu_1 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu_2 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu_3 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu_4 = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()

    def forward(self, l):
        x = l
        x1 = self.leaky_relu_1(self.conv1(x))
        x2 = self.leaky_relu_2(self.conv2(x1))
        x3 = self.leaky_relu_3(self.conv3(x2))
        x4 = self.leaky_relu_4(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        return x5


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'opts': _mock_config()}]
