import torch
import torch.utils.data
from torch import nn


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
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.
            calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class AttentionConditioningLayer(nn.Module):
    """Adapted from the LocationLayer in https://github.com/NVIDIA/tacotron2/blob/master/model.py
    1D Conv model over a concatenation of the previous attention and the accumulated attention values
    """

    def __init__(self, input_dim=2, attention_n_filters=32,
        attention_kernel_sizes=[5, 3], attention_dim=640):
        super(AttentionConditioningLayer, self).__init__()
        self.location_conv_hidden = ConvNorm(input_dim, attention_n_filters,
            kernel_size=attention_kernel_sizes[0], padding=None, bias=True,
            stride=1, dilation=1, w_init_gain='relu')
        self.location_conv_out = ConvNorm(attention_n_filters,
            attention_dim, kernel_size=attention_kernel_sizes[1], padding=
            None, bias=True, stride=1, dilation=1, w_init_gain='sigmoid')
        self.conv_layers = nn.Sequential(self.location_conv_hidden, nn.ReLU
            (), self.location_conv_out, nn.Sigmoid())

    def forward(self, attention_weights_cat):
        return self.conv_layers(attention_weights_cat)


def get_inputs():
    return [torch.rand([4, 2, 64])]


def get_init_inputs():
    return [[], {}]
