import torch
import torch.nn as nn
from functools import partial


def ncsn_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1,
    init_scale=1.0, padding=1):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv = nn.Conv2d(in_planes, out_planes, stride=stride, bias=bias,
        dilation=dilation, padding=padding, kernel_size=3)
    conv.weight.data *= init_scale
    if bias:
        conv.bias.data *= init_scale
    return conv


def ncsn_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1,
    init_scale=1.0, padding=0):
    """1x1 convolution. Same as NCSNv1/v2."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=bias, dilation=dilation, padding=padding)
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


class ConvMeanPool(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True,
        adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1,
                padding=kernel_size // 2, bias=biases)
            self.conv = conv
        else:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1,
                padding=kernel_size // 2, bias=biases)
            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)), conv)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
            output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.0
        return output


class ResidualBlock(nn.Module):

    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
        normalization=nn.InstanceNorm2d, adjust_padding=False, dilation=1):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=
                    dilation)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=
                    dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3,
                    adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1,
                    adjust_padding=adjust_padding)
        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=
                    dilation)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=
                    dilation)
            else:
                conv_shortcut = partial(ncsn_conv1x1)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception('invalid resample value')
        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)
        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)
        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        return shortcut + output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
