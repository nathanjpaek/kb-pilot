import torch
import torch.utils.data
import torch.nn.functional as F


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=None, dilation=1, bias=True, w_init_gain='linear', dropout=0.0
        ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.dropout = dropout
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.
            calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        if self.training and self.dropout > 0.0:
            conv_signal = F.dropout(conv_signal, p=self.dropout)
        return conv_signal


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
