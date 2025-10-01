import torch
import torch.nn as nn


class SuperSimpleSemSegNet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3,
            padding=1, stride=1)
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]
