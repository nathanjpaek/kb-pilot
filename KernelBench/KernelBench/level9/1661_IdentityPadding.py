import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityPadding(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_filters': 4, 'channels_in': 4, 'stride': 1}]
