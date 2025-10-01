import torch
from abc import *
import torch.nn.functional as F
from torch.optim import *


def orthogonal_init(layer, nonlinearity='relu'):
    if isinstance(nonlinearity, str):
        if nonlinearity == 'policy':
            gain = 0.01
        else:
            gain = torch.nn.init.calculate_gain(nonlinearity)
    else:
        gain = nonlinearity
    if isinstance(layer, list):
        for l in layer:
            torch.nn.init.orthogonal_(l.weight.data, gain)
            torch.nn.init.zeros_(l.bias.data)
    else:
        torch.nn.init.orthogonal_(layer.weight.data, gain)
        torch.nn.init.zeros_(layer.bias.data)


class MLP(torch.nn.Module):

    def __init__(self, D_in, D_hidden=512):
        super(MLP, self).__init__()
        self.l = torch.nn.Linear(D_in, D_hidden)
        self.D_head_out = D_hidden
        for layer in self.__dict__['_modules'].values():
            orthogonal_init(layer)

    def forward(self, x):
        x = F.relu(self.l(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4}]
