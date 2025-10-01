import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvToVector(nn.Module):

    def __init__(self, in_channels, padding=1):
        super(ConvToVector, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, padding=padding)
        self.conv3 = nn.Conv2d(6, 12, kernel_size=3, padding=padding)
        self.conv8 = nn.Conv2d(12, 6, kernel_size=3, padding=0)
        self.conv9 = nn.Conv2d(6, 3, kernel_size=3, padding=0)
        self.conv10 = nn.Conv2d(3, 1, kernel_size=3, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
