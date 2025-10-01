from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k,
        stride=s, padding=p, dilation=dilation, groups=groups)


class Adjust_naive(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.conv1 = get_conv2d_layer(in_c=2, out_c=32, k=5, s=1, p=2)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=1, k=5, s=1, p=2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, l, alpha):
        input = torch.cat([l, alpha], dim=1)
        x = self.conv1(input)
        x = self.conv2(self.leaky_relu(x))
        x = self.conv3(self.leaky_relu(x))
        x = self.conv4(self.leaky_relu(x))
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config()}]
