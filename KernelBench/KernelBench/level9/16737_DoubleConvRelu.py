import torch
from torch import nn
from torch.nn import functional as F


class DoubleConvRelu(nn.Module):

    def __init__(self, in_dec_filters: 'int', out_filters: 'int'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3,
            padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3,
            padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dec_filters': 4, 'out_filters': 4}]
