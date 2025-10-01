import torch
from torch import nn
from torch.nn import functional as F


class ConvTran(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvTran, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1)
        self.batch_norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        h = self.conv_t(x)
        h = self.batch_norm(h)
        return F.relu(h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
