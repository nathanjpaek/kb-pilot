import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import tee


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LinearNet(nn.Module):

    def __init__(self, layers, activation=torch.nn.ELU, layer_norm=False,
        linear_layer=nn.Linear):
        super(LinearNet, self).__init__()
        self.input_shape = layers[0]
        self.output_shape = layers[-1]
        if layer_norm:

            def layer_fn(layer):
                return [('linear_{}'.format(layer[0]), linear_layer(layer[1
                    ][0], layer[1][1])), ('layer_norm_{}'.format(layer[0]),
                    LayerNorm(layer[1][1])), ('act_{}'.format(layer[0]),
                    activation())]
        else:

            def layer_fn(layer):
                return [('linear_{}'.format(layer[0]), linear_layer(layer[1
                    ][0], layer[1][1])), ('act_{}'.format(layer[0]),
                    activation())]
        self.net = torch.nn.Sequential(OrderedDict([x for y in map(lambda
            layer: layer_fn(layer), enumerate(pairwise(layers))) for x in y]))

    def forward(self, x):
        x = self.net.forward(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'layers': [4, 4]}]
