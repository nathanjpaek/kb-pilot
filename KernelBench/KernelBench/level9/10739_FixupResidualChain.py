import torch
import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
from collections import OrderedDict


def _get_activation(activation):
    valid = ['relu', 'leaky_relu', 'lrelu', 'tanh', 'sigmoid']
    assert activation in valid, 'activation should be one of {}'.format(valid)
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    if activation == 'leaky_relu' or activation == 'lrelu':
        return nn.LeakyReLU(inplace=True)
    if activation == 'sigmoid':
        return nn.Sigmoid()
    if activation == 'tanh':
        return nn.Tanh()
    return None


def _init_fc_or_conv(fc_conv, activation):
    gain = 1.0
    if activation is not None:
        gain = nn.init.calculate_gain(activation)
    nn.init.xavier_uniform_(fc_conv.weight, gain)
    if fc_conv.bias is not None:
        nn.init.constant_(fc_conv.bias, 0.0)


def _get_norm_layer(norm_layer, channels):
    valid = ['instance', 'batch']
    assert norm_layer in valid, 'norm_layer should be one of {}'.format(valid)
    if norm_layer == 'instance':
        layer = nn.InstanceNorm2d(channels, affine=True)
    elif norm_layer == 'batch':
        layer = nn.BatchNorm2d(channels, affine=True)
    nn.init.constant_(layer.bias, 0.0)
    nn.init.constant_(layer.weight, 1.0)
    return layer


class ConvModule(nn.Module):
    """Basic convolution module with conv + norm(optional) + activation(optional).

    Args:
      n_in(int): number of input channels.
      n_out(int): number of output channels.
      ksize(int): size of the convolution kernel (square).
      stride(int): downsampling factor
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """

    def __init__(self, n_in, n_out, ksize=3, stride=1, pad=True, activation
        =None, norm_layer=None):
        super(ConvModule, self).__init__()
        assert isinstance(n_in, int
            ) and n_in > 0, 'Input channels should be a positive integer got {}'.format(
            n_in)
        assert isinstance(n_out, int
            ) and n_out > 0, 'Output channels should be a positive integer got {}'.format(
            n_out)
        assert isinstance(ksize, int
            ) and ksize > 0, 'Kernel size should be a positive integer got {}'.format(
            ksize)
        padding = (ksize - 1) // 2 if pad else 0
        use_bias_in_conv = norm_layer is None
        self.add_module('conv', nn.Conv2d(n_in, n_out, ksize, stride=stride,
            padding=padding, bias=use_bias_in_conv))
        if norm_layer is not None:
            self.add_module('norm', _get_norm_layer(norm_layer, n_out))
        if activation is not None:
            self.add_module('activation', _get_activation(activation))
        _init_fc_or_conv(self.conv, activation)

    def forward(self, x):
        for c in self.children():
            x = c(x)
        return x


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, n_features, ksize=3, pad=True, activation='relu'):
        super(FixupBasicBlock, self).__init__()
        self.bias1a = nn.Parameter(th.zeros(1))
        self.conv1 = ConvModule(n_features, n_features, ksize=ksize, stride
            =1, pad=pad, activation=None, norm_layer=None)
        self.bias1b = nn.Parameter(th.zeros(1))
        self.activation = _get_activation(activation)
        self.bias2a = nn.Parameter(th.zeros(1))
        self.conv2 = ConvModule(n_features, n_features, ksize=ksize, stride
            =1, pad=pad, activation=None, norm_layer=None)
        self.scale = nn.Parameter(th.ones(1))
        self.bias2b = nn.Parameter(th.zeros(1))
        self.activation2 = _get_activation(activation)
        self.ksize = 3
        self.pad = pad

    def forward(self, x):
        identity = x
        out = self.conv1(x + self.bias1a)
        out = self.activation(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b
        crop = (self.ksize - 1) // 2 * 2
        if crop > 0 and not self.pad:
            identity = identity[:, :, crop:-crop, crop:-crop]
        out += identity
        out = self.activation2(out)
        return out


class FixupResidualChain(nn.Module):
    """Linear chain of residual blocks.

    Args:
      n_features(int): number of input channels.
      depth(int): number of residual blocks
      ksize(int): size of the convolution kernel (square).
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
      pad(bool): if True, zero pad the convs to maintain a constant size.
    """

    def __init__(self, n_features, depth=3, ksize=3, activation='relu',
        norm_layer=None, pad=True):
        super(FixupResidualChain, self).__init__()
        assert isinstance(n_features, int
            ) and n_features > 0, 'Number of feature channels should be a positive integer'
        assert isinstance(ksize, int) and ksize > 0 or isinstance(ksize, list
            ), 'Kernel size should be a positive integer or a list of integers'
        assert isinstance(depth, int
            ) and depth > 0 and depth < 16, 'Depth should be a positive integer lower than 16'
        self.depth = depth
        layers = OrderedDict()
        for lvl in range(depth):
            blockname = 'resblock{}'.format(lvl)
            layers[blockname] = FixupBasicBlock(n_features, ksize=ksize,
                activation=activation, pad=pad)
        self.net = nn.Sequential(layers)
        self._reset_weights()

    def _reset_weights(self):
        for m in self.net.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.conv.weight, mean=0, std=np.sqrt(2 /
                    (m.conv1.conv.weight.shape[0] * np.prod(m.conv1.conv.
                    weight.shape[2:]))) * self.depth ** -0.5)
                nn.init.constant_(m.conv2.conv.weight, 0)

    def forward(self, x):
        x = self.net(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
