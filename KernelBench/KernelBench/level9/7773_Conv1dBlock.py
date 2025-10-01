import torch
from torch import nn


class Conv1dBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=
        0, norm='none', activation='relu', pad_type='zero'):
        super(Conv1dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad1d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad1d(padding)
        elif pad_type == 'zero':
            self.pad = None
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        if pad_type == 'zero':
            self.conv = nn.Conv1d(input_dim, output_dim, kernel_size,
                stride, padding, bias=self.use_bias)
        else:
            self.conv = nn.Conv1d(input_dim, output_dim, kernel_size,
                stride, bias=self.use_bias)

    @staticmethod
    def calc_samepad_size(input_dim, kernel_size, stride, dilation=1):
        return ((input_dim - 1) * stride - input_dim + kernel_size + (
            kernel_size - 1) * (dilation - 1)) / 2

    def forward(self, x):
        if self.pad:
            x = self.pad(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'kernel_size': 4,
        'stride': 1}]
