import torch
from torch import nn as nn
import torch.utils.data.distributed


def get_activation_function(activation: 'str') ->nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == 'Linear':
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, activation='PReLU', dropout=0.1,
        d_out=None):
        """Initialization.

        :param d_model: the input dimension.
        :param d_ff: the hidden dimension.
        :param activation: the activation function.
        :param dropout: the dropout rate.
        :param d_out: the output dimension, the default value is equal to d_model.
        """
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None:
            d_out = d_model
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.act_func = get_activation_function(activation)

    def forward(self, x):
        """
        The forward function
        :param x: input tensor.
        :return:
        """
        return self.W_2(self.dropout(self.act_func(self.W_1(x))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_ff': 4}]
