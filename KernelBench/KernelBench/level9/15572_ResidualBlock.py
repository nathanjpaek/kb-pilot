import math
import torch
import torch.nn as nn


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ResidualBlock(nn.Module):
    """ Residual Block """

    def __init__(self, d_encoder, residual_channels, dropout):
        super(ResidualBlock, self).__init__()
        self.conv_layer = ConvNorm(residual_channels, 2 * residual_channels,
            kernel_size=3, stride=1, padding=int((3 - 1) / 2), dilation=1)
        self.diffusion_projection = LinearNorm(residual_channels,
            residual_channels)
        self.conditioner_projection = ConvNorm(d_encoder, 2 *
            residual_channels, kernel_size=1)
        self.output_projection = ConvNorm(residual_channels, 2 *
            residual_channels, kernel_size=1)

    def forward(self, x, conditioner, diffusion_step, mask=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1
            )
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.conv_layer(y) + conditioner
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_encoder': 4, 'residual_channels': 4, 'dropout': 0.5}]
