import torch
from torch import nn


class CausualConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=1, dilation=1, bias=True, w_init_gain='linear', param=None):
        super(CausualConv, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2) * 2
        else:
            self.padding = padding * 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=self.padding, dilation=
            dilation, bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.
            calculate_gain(w_init_gain, param=param))

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.padding]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
