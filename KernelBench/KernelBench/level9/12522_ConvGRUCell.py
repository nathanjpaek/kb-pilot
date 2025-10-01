import torch
from torch import nn
import torch.nn.functional as F


def one_param(m):
    """First parameter in `m`"""
    return next(m.parameters())


class ConvGRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), bias=True,
        activation=F.tanh, batchnorm=False):
        """
        Initialize ConvGRU cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)
            ) else [kernel_size] * 2
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias = bias
        self.activation = activation
        self.batchnorm = batchnorm
        self.conv_zr = nn.Conv2d(in_channels=self.input_dim + self.
            hidden_dim, out_channels=2 * self.hidden_dim, kernel_size=self.
            kernel_size, padding=self.padding, bias=self.bias)
        self.conv_h1 = nn.Conv2d(in_channels=self.input_dim, out_channels=
            self.hidden_dim, kernel_size=self.kernel_size, padding=self.
            padding, bias=self.bias)
        self.conv_h2 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=
            self.hidden_dim, kernel_size=self.kernel_size, padding=self.
            padding, bias=self.bias)
        self.reset_parameters()

    def forward(self, input, h_prev=None):
        if h_prev is None:
            h_prev = self.init_hidden(input)
        combined = torch.cat((input, h_prev), dim=1)
        combined_conv = F.sigmoid(self.conv_zr(combined))
        z, r = torch.split(combined_conv, self.hidden_dim, dim=1)
        h_ = self.activation(self.conv_h1(input) + r * self.conv_h2(h_prev))
        h_cur = (1 - z) * h_ + z * h_prev
        return h_cur

    def init_hidden(self, input):
        bs, _ch, h, w = input.shape
        return one_param(self).new_zeros(bs, self.hidden_dim, h, w)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv_zr.weight, gain=nn.init.
            calculate_gain('tanh'))
        self.conv_zr.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h1.weight, gain=nn.init.
            calculate_gain('tanh'))
        self.conv_h1.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h2.weight, gain=nn.init.
            calculate_gain('tanh'))
        self.conv_h2.bias.data.zero_()
        if self.batchnorm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4}]
