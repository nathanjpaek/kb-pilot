import torch
import torch.nn as nn
import torch.utils.data


class Conv1d(nn.Conv1d):
    """
    :param in_channels: Scalar
    :param out_channels: Scalar
    :param kernel_size: Scalar
    :param activation_fn: activation function
    :param drop_rate: Scalar. dropout rate
    :param stride: Scalar
    :param padding: padding type
    :param dilation: Scalar
    :param groups: Scalar
    :param bias: Boolean.
    :param bn: Boolean. whether it uses batch normalization

    """

    def __init__(self, in_channels, out_channels, kernel_size,
        activation_fn=None, drop_rate=0.0, stride=1, padding='same',
        dilation=1, groups=1, bias=True, bn=False):
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate
        if padding == 'same':
            padding = kernel_size // 2 * dilation
            self.even_kernel = not bool(kernel_size % 2)
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias)
        self.drop_out = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=
            0.001) if bn else None

    def forward(self, x):
        """
        :param x: (N, C_in, T) Tensor.

        Returns:
            y: (N, C_out, T) Tensor.

        """
        y = super(Conv1d, self).forward(x)
        y = self.batch_norm(y) if self.batch_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        y = self.drop_out(y) if self.drop_out is not None else y
        y = y[:, :, :-1] if self.even_kernel else y
        return y


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
