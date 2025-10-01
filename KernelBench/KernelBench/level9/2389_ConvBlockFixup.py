import torch
from torch import nn


class ConvBlockFixup(nn.Module):

    def __init__(self, filter_width, input_filters, nb_filters, dilation):
        super(ConvBlockFixup, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.
            filter_width, 1), dilation=(self.dilation, 1), bias=False,
            padding='same')
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.
            filter_width, 1), dilation=(self.dilation, 1), bias=False,
            padding='same')
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x
        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b
        out += identity
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'filter_width': 4, 'input_filters': 4, 'nb_filters': 4,
        'dilation': 1}]
