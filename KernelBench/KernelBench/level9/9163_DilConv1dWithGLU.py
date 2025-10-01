import torch
import torch.nn as nn
import torch.nn.functional as F


class DilConv1dWithGLU(nn.Module):

    def __init__(self, num_channels, dilation, lenght=100, kernel_size=2,
        activation=F.leaky_relu, residual_connection=True, dropout=0.2):
        super(DilConv1dWithGLU, self).__init__()
        self.dilation = dilation
        self.start_ln = nn.LayerNorm(num_channels)
        self.start_conv1x1 = nn.Conv1d(num_channels, num_channels,
            kernel_size=1)
        self.dilconv_ln = nn.LayerNorm(num_channels)
        self.dilated_conv = nn.Conv1d(num_channels, num_channels, dilation=
            dilation, kernel_size=kernel_size, padding=dilation)
        self.gate_ln = nn.LayerNorm(num_channels)
        self.end_conv1x1 = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.gated_conv1x1 = nn.Conv1d(num_channels, num_channels,
            kernel_size=1)
        self.activation = activation
        self.buffer = None
        self.residual_connection = residual_connection

    def clear_buffer(self):
        self.buffer = None

    def forward(self, x_inp, sampling=False):
        x = self.start_ln(x_inp.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.start_conv1x1(x)
        x = self.dilconv_ln(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        if sampling:
            if self.buffer is None:
                self.buffer = x
            else:
                pre_buffer = torch.cat([self.buffer, x], dim=2)
                self.buffer = pre_buffer[:, :, -(self.dilation + 1):]
            if self.buffer.shape[2] == self.dilation + 1:
                x = self.buffer
            else:
                x = torch.cat([torch.zeros(self.buffer.shape[0], self.
                    buffer.shape[1], self.dilation + 1 - self.buffer.shape[
                    2], device=x_inp.device), self.buffer], dim=2)
            x = self.dilated_conv(x)[:, :, self.dilation:]
            x = x[:, :, :x_inp.shape[-1]]
        else:
            x = self.dilated_conv(x)[:, :, :x_inp.shape[-1]]
        x = self.gate_ln(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.end_conv1x1(x) * torch.sigmoid(self.gated_conv1x1(x))
        if self.residual_connection:
            x = x + x_inp
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4, 'dilation': 1}]
