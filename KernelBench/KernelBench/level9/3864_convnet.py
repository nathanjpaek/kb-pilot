import torch
import torch.nn as nn


class convnet(nn.Module):

    def __init__(self, in_channel, dim):
        super(convnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'dim': 4}]
