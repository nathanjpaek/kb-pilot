import torch
import torch.nn as nn
import torch.utils


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/    0344c5503ee55e24f0de7f37336a6e08f10976fd/    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, depthwise=False, bias=True,
        width_mult_list=[1.0]):
        super(USConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult_list = width_mult_list
        self.ratio = 1.0, 1.0

    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input):
        assert self.ratio[0] in self.width_mult_list, str(self.ratio[0]
            ) + ' in? ' + str(self.width_mult_list)
        self.in_channels = make_divisible(self.in_channels_max * self.ratio[0])
        assert self.ratio[1] in self.width_mult_list, str(self.ratio[1]
            ) + ' in? ' + str(self.width_mult_list)
        self.out_channels = make_divisible(self.out_channels_max * self.
            ratio[1])
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.
            padding, self.dilation, self.groups)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
