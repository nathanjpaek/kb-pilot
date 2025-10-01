import torch
import torch.nn as nn
from math import sqrt as sqrt


class SeqExpandConv(nn.Module):

    def __init__(self, in_channels, out_channels, seq_length):
        super(SeqExpandConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1,
            1), padding=(1, 0, 0), bias=False)
        self.seq_length = seq_length

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        x = x.view(batch_size // self.seq_length, self.seq_length,
            in_channels, height, width)
        x = self.conv(x.transpose(1, 2).contiguous()).transpose(2, 1
            ).contiguous()
        x = x.flatten(0, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'seq_length': 4}]
