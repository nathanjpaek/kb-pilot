import torch
import torch.nn as nn
import torch.nn.functional as F


class Unit1D(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_shape=1, stride
        =1, padding='same', activation_fn=F.relu, use_bias=True):
        super(Unit1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, output_channels, kernel_shape,
            stride, padding=0, bias=use_bias)
        self._activation_fn = activation_fn
        self._padding = padding
        self._stride = stride
        self._kernel_shape = kernel_shape

    def compute_pad(self, t):
        if t % self._stride == 0:
            return max(self._kernel_shape - self._stride, 0)
        else:
            return max(self._kernel_shape - t % self._stride, 0)

    def forward(self, x):
        if self._padding == 'same':
            _batch, _channel, t = x.size()
            pad_t = self.compute_pad(t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            x = F.pad(x, [pad_t_f, pad_t_b])
        x = self.conv1d(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'output_channels': 4}]
