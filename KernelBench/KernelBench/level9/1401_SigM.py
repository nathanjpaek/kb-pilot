import torch
import torch.utils.data
import torch.nn as nn


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class SigM(nn.Module):

    def __init__(self, in_channel, output_channel, reduction=1):
        super(SigM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output_channel = output_channel
        self.h_sigmoid = h_sigmoid()
        if in_channel == output_channel:
            self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        else:
            self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv2d(
                in_channel, output_channel, kernel_size=1, stride=1,
                padding=0), nn.ReLU(inplace=True))

    def forward(self, x):
        x_sz = len(x.size())
        if x_sz == 2:
            x = x.unsqueeze(-1)
        b, _c, _ = x.size()
        y = self.fc(x).view(b, self.output_channel, 1)
        y = self.h_sigmoid(y)
        out = x * y.expand_as(x)
        if x_sz == 2:
            out = out.squeeze(-1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'output_channel': 4}]
