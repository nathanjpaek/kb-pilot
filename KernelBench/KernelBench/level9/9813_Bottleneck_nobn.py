import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck_nobn(nn.Module):

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck_nobn, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1,
            bias=False)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        out = torch.cat([out, x], 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'growth_rate': 4}]
