import torch
import torch.nn as nn


class Gated_Conv_1d(nn.Module):

    def __init__(self, channels, kernel_size, stride=1, padding=0, dilation
        =1, groups=1, bias=True):
        super(Gated_Conv_1d, self).__init__()
        self.dilation = dilation
        self.channels = channels
        self.conv_dil = nn.Conv1d(in_channels=channels, out_channels=2 *
            channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, groups=groups, bias=bias)
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_dil(x)
        tn, sg = torch.split(x, self.channels, 1)
        return self.tan(tn) * self.sig(sg)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'kernel_size': 4}]
