import torch
import torch.utils.data
import torch
import torch.nn as nn


class Conv2dAct(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=1, activation='relu'):
        super(Conv2dAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class VarianceC(nn.Module):

    def __init__(self):
        super(VarianceC, self).__init__()

    def forward(self, x):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        sub_x = x.sub(mean_x)
        x = torch.mean(torch.mul(sub_x, sub_x), dim=1, keepdim=True)
        return x


class ScoringFunction(nn.Module):

    def __init__(self, in_channels, var=False):
        super(ScoringFunction, self).__init__()
        if var:
            self.reduce_channel = VarianceC()
        else:
            self.reduce_channel = Conv2dAct(in_channels, 1, 1, 'sigmoid')

    def forward(self, x):
        x = self.reduce_channel(x)
        x = x.view(x.size(0), -1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
