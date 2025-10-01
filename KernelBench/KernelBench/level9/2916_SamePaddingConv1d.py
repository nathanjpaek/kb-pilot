import torch
import torch.nn as nn


class SamePaddingConv1d(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=int((
            kernel_size - 1) / 2))

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'kernel_size': 4}]
