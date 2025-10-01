import torch
from torch import nn
import torch.nn.functional as F


class aeEncoder(nn.Module):

    def __init__(self, capacity, channel):
        super(aeEncoder, self).__init__()
        self.c = capacity
        self.channel = channel
        self.conv1 = nn.Conv2d(in_channels=self.channel, out_channels=self.
            c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c * 2,
            kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.c * 2, out_channels=self.c *
            4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.c * 4, out_channels=self.c *
            8, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=self.c * 8, out_channels=self.c *
            16, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'capacity': 4, 'channel': 4}]
