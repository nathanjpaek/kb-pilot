import torch
import torch.nn as nn


class AgentConvBlock(nn.Module):

    def __init__(self, nin, nout, ksize=3):
        super(AgentConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(nin, nout, ksize, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nout, nout, ksize, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        h = self.lrelu1(self.conv1(x))
        h = self.lrelu2(self.conv2(h))
        return self.pool(h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nin': 4, 'nout': 4}]
