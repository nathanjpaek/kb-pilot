import torch
from torch import nn
import torch as t


class Conv(nn.Module):
    """Convolution Module."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=True, w_init='linear'):
        """init."""
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.
            calculate_gain(w_init))

    def forward(self, x):
        """forward."""
        x = self.conv(x)
        return x


class FFN(nn.Module):
    """Positionwise Feed-Forward Network."""

    def __init__(self, num_hidden):
        """:param num_hidden: dimension of hidden."""
        super(FFN, self).__init__()
        self.w_1 = Conv(num_hidden, num_hidden * 4, kernel_size=1, w_init=
            'relu')
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        """forward."""
        x = input_.transpose(1, 2)
        x = self.w_2(t.relu(self.w_1(x)))
        x = x.transpose(1, 2)
        x = x + input_
        x = self.layer_norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_hidden': 4}]
