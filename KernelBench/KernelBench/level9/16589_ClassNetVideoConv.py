import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Unit3D(nn.Module):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1),
        stride=(1, 1, 1), padding=0, activation_fn=F.relu, use_batch_norm=
        True, use_bias=False, name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self.
            _output_channels, kernel_size=self._kernel_shape, stride=self.
            _stride, padding=0, bias=self._use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001,
                momentum=0.01)

    def compute_pad(self, dim, s):
        """Get the zero padding number."""
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - s % self._stride[dim], 0)

    def forward(self, x):
        """
        Connects the module to inputs. Dynamically pad based on input size in forward function.
        Args:
            x: Inputs to the Unit3D component.

        Returns:
            Outputs from the module.
        """
        _batch, _channel, time, height, width = x.size()
        pad_t = self.compute_pad(0, time)
        pad_h = self.compute_pad(1, height)
        pad_w = self.compute_pad(2, width)
        pad_t_front = pad_t // 2
        pad_t_back = pad_t - pad_t_front
        pad_h_front = pad_h // 2
        pad_h_back = pad_h - pad_h_front
        pad_w_front = pad_w // 2
        pad_w_back = pad_w - pad_w_front
        pad = (pad_w_front, pad_w_back, pad_h_front, pad_h_back,
            pad_t_front, pad_t_back)
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class ClassNetVideoConv(nn.Module):
    """Classifier network for video input refer to MMSADA.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 1024.
        n_class (int, optional): the number of classes. Defaults to 8.

    References:
        Munro Jonathan, and Dima Damen. "Multi-modal domain adaptation for fine-grained action recognition."
        In CVPR, pp. 122-132. 2020.
    """

    def __init__(self, input_size=1024, n_class=8):
        super(ClassNetVideoConv, self).__init__()
        self.dp = nn.Dropout()
        self.logits = Unit3D(in_channels=input_size, output_channels=
            n_class, kernel_shape=[1, 1, 1], padding=0, activation_fn=None,
            use_batch_norm=False, use_bias=True)

    def forward(self, input):
        x = self.logits(self.dp(input))
        return x


def get_inputs():
    return [torch.rand([4, 1024, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
