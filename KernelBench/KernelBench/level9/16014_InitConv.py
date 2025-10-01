import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class InitConv(nn.Module):

    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3,
            padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
