import torch
import torch.nn as nn
import torch.utils.data


class Conv(nn.Module):

    def __init__(self, filters0, filters1, kernel_size, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1,
            padding=kernel_size // 2, bias=bias)
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class Head(nn.Module):

    def __init__(self, input_size, out_filters, outputs):
        super().__init__()
        self.board_size = input_size[1] * input_size[2]
        self.out_filters = out_filters
        self.conv = Conv(input_size[0], out_filters, 1, bn=False)
        self.activation = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(self.board_size * out_filters, outputs, bias=False)

    def forward(self, x):
        h = self.activation(self.conv(x))
        h = self.fc(h.view(-1, self.board_size * self.out_filters))
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': [4, 4, 4], 'out_filters': 4, 'outputs': 4}]
