import torch
from collections import OrderedDict
import torch.nn as nn


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.
            calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Stoplinear(nn.Module):
    """
    Adding hidden layer to stopplinear to improve it
    """

    def __init__(self, input_size, p=0.1):
        super(Stoplinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = input_size // 3
        self.layers = nn.Sequential(OrderedDict([('fc1', Linear(self.
            input_size, self.hidden_size, w_init='sigmoid')), ('activation',
            nn.ReLU()), ('dropout', nn.Dropout(p)), ('fc2', Linear(self.
            hidden_size, 1, w_init='sigmoid'))]))

    def forward(self, input_):
        out = self.layers(input_)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
