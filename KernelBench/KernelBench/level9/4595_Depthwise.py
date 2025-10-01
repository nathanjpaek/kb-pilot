import torch
import torch.nn as nn
import torch.nn.functional as F


class Depthwise(nn.Module):

    def __init__(self, Cin=10, K=3, depth_multiplier=1):
        super(Depthwise, self).__init__()
        self.conv1 = nn.Conv2d(Cin, depth_multiplier * Cin, kernel_size=K,
            groups=Cin, bias=False, padding=0, stride=1)

    def forward(self, x):
        return F.relu(self.conv1(x))


def get_inputs():
    return [torch.rand([4, 10, 64, 64])]


def get_init_inputs():
    return [[], {}]
