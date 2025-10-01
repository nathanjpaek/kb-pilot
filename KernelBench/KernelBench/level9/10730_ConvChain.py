import torch
import torch.utils.data
import torch.nn as nn


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


class ConvChain(nn.Module):
    """Linear chain of convolution layers.

    Args:
      n_in(int): number of input channels.
      ksize(int or list of int): size of the convolution kernel (square).
      width(int or list of int): number of features channels in the intermediate layers.
      depth(int): number of layers
      strides(list of int): stride between kernels. If None, defaults to 1 for all.
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """

    def __init__(self, n_in, ksize=3, width=64, depth=3, strides=None, pad=
        True, activation='relu', norm_layer=None):
        super(ConvChain, self).__init__()
        assert isinstance(n_in, int
            ) and n_in > 0, 'Input channels should be a positive integer'
        assert isinstance(ksize, int) and ksize > 0 or isinstance(ksize, list
            ), 'Kernel size should be a positive integer or a list of integers'
        assert isinstance(depth, int
            ) and depth > 0, 'Depth should be a positive integer'
        assert isinstance(width, int) or isinstance(width, list
            ), 'Width should be a list or an int'
        _in = [n_in]
        if strides is None:
            _strides = [1] * depth
        else:
            assert isinstance(strides, list), 'strides should be a list'
            assert len(strides
                ) == depth, 'strides should have `depth` elements'
            _strides = strides
        if isinstance(width, int):
            _in = _in + [width] * (depth - 1)
            _out = [width] * depth
        elif isinstance(width, list):
            assert len(width
                ) == depth, 'Specifying width with a list should have `depth` elements'
            _in = _in + width[:-1]
            _out = width
        if isinstance(ksize, int):
            _ksizes = [ksize] * depth
        elif isinstance(ksize, list):
            assert len(ksize
                ) == depth, "kernel size list should have 'depth' entries"
            _ksizes = ksize
        _activations = [activation] * depth
        _norms = [norm_layer] * depth
        for lvl in range(depth):
            self.add_module('conv{}'.format(lvl), ConvModule(_in[lvl], _out
                [lvl], _ksizes[lvl], stride=_strides[lvl], pad=pad,
                activation=_activations[lvl], norm_layer=_norms[lvl]))

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4}]
