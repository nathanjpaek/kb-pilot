import torch
import torch.nn as nn
import torch.nn.functional as F


class Unit3D(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1),
        stride=(1, 1, 1), padding='spatial_valid', activation_fn=F.relu,
        use_batch_norm=False, use_bias=False):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.padding = padding
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001,
                momentum=0.01)
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self.
            _output_channels, kernel_size=self._kernel_shape, stride=self.
            _stride, padding=0, bias=self._use_bias)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - s % self._stride[dim], 0)

    def forward(self, x):
        if self.padding == 'same':
            _batch, _channel, t, h, w = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f
            pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
            x = F.pad(x, pad)
        if self.padding == 'spatial_valid':
            _batch, _channel, t, h, w = x.size()
            pad_t = self.compute_pad(0, t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad = [0, 0, 0, 0, pad_t_f, pad_t_b]
            x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'output_channels': 4}]
