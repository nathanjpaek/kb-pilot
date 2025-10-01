import torch
import torch.utils.data
import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, init_gain='linear'
        ):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.
            calculate_gain(init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LocationLayer(nn.Module):

    def __init__(self, attention_dim, attention_n_filters=32,
        attention_kernel_size=31):
        super(LocationLayer, self).__init__()
        self.location_conv = nn.Conv1d(in_channels=2, out_channels=
            attention_n_filters, kernel_size=attention_kernel_size, stride=
            1, padding=(attention_kernel_size - 1) // 2, bias=False)
        self.location_dense = Linear(attention_n_filters, attention_dim,
            bias=False, init_gain='tanh')

    def forward(self, attention_cat):
        processed_attention = self.location_conv(attention_cat)
        processed_attention = self.location_dense(processed_attention.
            transpose(1, 2))
        return processed_attention


def get_inputs():
    return [torch.rand([4, 2, 64])]


def get_init_inputs():
    return [[], {'attention_dim': 4}]
