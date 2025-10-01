import torch
import torch.nn as nn
import torch.nn.functional as F


class TransposedConv3d(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_shape=(3, 3, 3),
        stride=(2, 1, 1), padding=(1, 1, 1), output_padding=(1, 0, 0),
        activation_fn=F.relu, use_batch_norm=False, use_bias=True):
        super(TransposedConv3d, self).__init__()
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self.transposed_conv3d = nn.ConvTranspose3d(in_channels,
            output_channels, kernel_shape, stride, padding=padding,
            output_padding=output_padding, bias=use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001,
                momentum=0.01)

    def forward(self, x):
        x = self.transposed_conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'output_channels': 4}]
