import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn


class Conv1D(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0,
        bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
            kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        return x.transpose(1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
