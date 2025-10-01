import torch
import torch.nn as nn
import torch.nn.functional as F


class encoderVH(nn.Module):

    def __init__(self):
        super(encoderVH, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=
            4, stride=2, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(4, 64)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128,
            kernel_size=4, stride=2, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(8, 128)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256,
            kernel_size=4, stride=2, padding=1, bias=True)
        self.gn3 = nn.GroupNorm(16, 256)

    def forward(self, x):
        x1 = F.relu(self.gn1(self.conv1(x)), True)
        x2 = F.relu(self.gn2(self.conv2(x1)), True)
        x3 = F.relu(self.gn3(self.conv3(x2)), True)
        return x3


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
