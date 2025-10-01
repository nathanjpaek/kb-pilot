import torch
import torch.nn as nn


class CausalConv1d(torch.nn.Conv1d):
    """Causal 1d convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=self.__padding,
            dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class ResidualBlock(nn.Module):
    """Residual block of WaveNet architecture"""

    def __init__(self, residual_channels, dilation_channels, skip_channels,
        n_globals, kernel_size=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.n_globals = n_globals
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.filter_conv = CausalConv1d(in_channels=self.residual_channels,
            out_channels=self.dilation_channels, kernel_size=self.
            kernel_size, dilation=self.dilation)
        self.gated_conv = CausalConv1d(in_channels=self.residual_channels,
            out_channels=self.dilation_channels, kernel_size=self.
            kernel_size, dilation=self.dilation)
        self.filter_linear = nn.Linear(in_features=self.n_globals,
            out_features=self.dilation_channels)
        self.gated_linear = nn.Linear(in_features=self.n_globals,
            out_features=self.dilation_channels)
        self._1x1_conv_res = nn.Conv1d(in_channels=self.dilation_channels,
            out_channels=self.residual_channels, kernel_size=1, dilation=1)
        self._1x1_conv_skip = nn.Conv1d(in_channels=self.dilation_channels,
            out_channels=self.skip_channels, kernel_size=1, dilation=1)

    def forward(self, x, h):
        h_f = self.filter_linear(h).unsqueeze(2)
        h_g = self.gated_linear(h).unsqueeze(2)
        x_f = self.filter_conv(x)
        x_g = self.gated_conv(x)
        z_f = torch.tanh(x_f + h_f)
        z_g = torch.sigmoid(x_g + h_g)
        z = torch.mul(z_f, z_g)
        skip = self._1x1_conv_skip(z)
        residual = x + self._1x1_conv_res(z)
        return skip, residual


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'residual_channels': 4, 'dilation_channels': 1,
        'skip_channels': 4, 'n_globals': 4}]
