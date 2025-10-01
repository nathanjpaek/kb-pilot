import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstNet(nn.Module):

    def __init__(self):
        super(FirstNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=
            3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
