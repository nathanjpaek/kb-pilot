import torch
import torch.utils.data
import torch
from torch import nn


class LateralBlock(nn.Module):

    def __init__(self, conv_dim, alpha):
        super(LateralBlock, self).__init__()
        self.conv = nn.Conv3d(conv_dim, conv_dim * 2, kernel_size=(5, 1, 1),
            stride=(alpha, 1, 1), padding=(2, 0, 0), bias=True)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        out = self.conv(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'conv_dim': 4, 'alpha': 4}]
