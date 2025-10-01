import math
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit
    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
        skip_out_channels=None, cin_channels=-1, gin_channels=-1, dropout=1 -
        0.95, padding=None, dilation=1, bias=True, *args, **kwargs):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(residual_channels, gate_channels, kernel_size,
            *args, padding=padding, dilation=dilation, bias=bias, **kwargs)
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = nn.Conv1d(gate_out_channels, residual_channels, 
            1, bias=bias)
        self.conv1x1_skip = nn.Conv1d(gate_out_channels, skip_out_channels,
            1, bias=bias)

    def forward(self, x):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
        Returns:
            Tensor: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        splitdim = 1
        x = self.conv(x)
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        x = torch.tanh(a) * torch.sigmoid(b)
        s = self.conv1x1_skip(x)
        x = self.conv1x1_out(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, s


def get_inputs():
    return [torch.rand([4, 4, 2])]


def get_init_inputs():
    return [[], {'residual_channels': 4, 'gate_channels': 4, 'kernel_size': 4}]
