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


class FCModule(nn.Module):
    """Basic fully connected module with optional dropout.

    Args:
      n_in(int): number of input channels.
      n_out(int): number of output channels.
      activation(str): nonlinear activation function.
      dropout(float): dropout ratio if defined, default to None: no dropout.
    """

    def __init__(self, n_in, n_out, activation=None, dropout=None):
        super(FCModule, self).__init__()
        assert isinstance(n_in, int
            ) and n_in > 0, 'Input channels should be a positive integer'
        assert isinstance(n_out, int
            ) and n_out > 0, 'Output channels should be a positive integer'
        self.add_module('fc', nn.Linear(n_in, n_out))
        if activation is not None:
            self.add_module('activation', _get_activation(activation))
        if dropout is not None:
            self.add_module('dropout', nn.Dropout(dropout, inplace=True))
        _init_fc_or_conv(self.fc, activation)

    def forward(self, x):
        for c in self.children():
            x = c(x)
        return x


class FCChain(nn.Module):
    """Linear chain of fully connected layers.

    Args:
      n_in(int): number of input channels.
      width(int or list of int): number of features channels in the intermediate layers.
      depth(int): number of layers
      activation(str): nonlinear activation function between convolutions.
      dropout(float or list of float): dropout ratio if defined, default to None: no dropout.
    """

    def __init__(self, n_in, width=64, depth=3, activation='relu', dropout=None
        ):
        super(FCChain, self).__init__()
        assert isinstance(n_in, int
            ) and n_in > 0, 'Input channels should be a positive integer'
        assert isinstance(depth, int
            ) and depth > 0, 'Depth should be a positive integer'
        assert isinstance(width, int) or isinstance(width, list
            ), 'Width should be a list or an int'
        _in = [n_in]
        if isinstance(width, int):
            _in = _in + [width] * (depth - 1)
            _out = [width] * depth
        elif isinstance(width, list):
            assert len(width
                ) == depth, 'Specifying width with a least: should have `depth` entries'
            _in = _in + width[:-1]
            _out = width
        _activations = [activation] * depth
        if dropout is not None:
            assert isinstance(dropout, float) or isinstance(dropout, list
                ), 'Dropout should be a float or a list of floats'
        if dropout is None or isinstance(dropout, float):
            _dropout = [dropout] * depth
        elif isinstance(dropout, list):
            assert len(dropout
                ) == depth, "When specifying a list of dropout, the list should have 'depth' elements."
            _dropout = dropout
        for lvl in range(depth):
            self.add_module('fc{}'.format(lvl), FCModule(_in[lvl], _out[lvl
                ], activation=_activations[lvl], dropout=_dropout[lvl]))

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4}]
