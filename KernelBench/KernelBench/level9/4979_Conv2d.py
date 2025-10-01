import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


def keep_variance_fn(x):
    return x + 0.001


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, keep_variance_fn=None,
        padding_mode='zeros'):
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
            stride, padding, dilation, False, _pair(0), groups, bias,
            padding_mode)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.conv2d(inputs_mean, self.weight, self.bias, self.
            stride, self.padding, self.dilation, self.groups)
        outputs_variance = F.conv2d(inputs_variance, self.weight ** 2, None,
            self.stride, self.padding, self.dilation, self.groups)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
