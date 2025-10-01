import torch
import torch.utils.data
from torch import nn


class LinearNorm(torch.nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias)

    def forward(self, signal):
        return self.conv(signal)


class LocationLayer(nn.Module):

    def __init__(self, attention_n_filters, attention_kernel_size,
        attention_dim):
        super(LocationLayer, self).__init__()
        self.location_conv = ConvNorm(1, attention_n_filters, kernel_size=
            attention_kernel_size, padding=int((attention_kernel_size - 1) /
            2), stride=1, dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
            bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cum):
        processed_attention_weights = self.location_conv(attention_weights_cum)
        processed_attention_weights = processed_attention_weights.transpose(
            1, 2)
        processed_attention_weights = self.location_dense(
            processed_attention_weights)
        return processed_attention_weights


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {'attention_n_filters': 4, 'attention_kernel_size': 4,
        'attention_dim': 4}]
