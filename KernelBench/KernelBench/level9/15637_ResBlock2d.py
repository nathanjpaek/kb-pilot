import torch
from torch import nn


class ResBlock2d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1)
        self.bn = nn.InstanceNorm2d(in_ch)
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm2d(out_ch)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        bypass = []
        if in_ch != out_ch:
            bypass.append(nn.Conv2d(in_ch, out_ch, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]
