import torch
from torch import nn


class ResBlock(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob, same='False'):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_chans)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_chans)
        self.same = same

    def forward(self, input):
        shortcuts = self.conv(input)
        if self.same == 'True':
            shortcuts = input
        out = self.conv1(input)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += shortcuts
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_chans': 4, 'out_chans': 4, 'drop_prob': 4}]
