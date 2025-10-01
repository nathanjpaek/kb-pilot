import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def _get_activation_fn(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    if activation == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'geglu':
        return GEGLU()


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class ContinuousEmbeddings(nn.Module):

    def __init__(self, n_cont_cols: 'int', embed_dim: 'int', activation:
        'str'=None, bias: 'bool'=True):
        super(ContinuousEmbeddings, self).__init__()
        self.n_cont_cols = n_cont_cols
        self.embed_dim = embed_dim
        self.activation = activation
        self.weight = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim)
            ) if bias else None
        self._reset_parameters()
        self.act_fn = _get_activation_fn(activation) if activation else None

    def _reset_parameters(self) ->None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: 'Tensor') ->Tensor:
        x = self.weight.unsqueeze(0) * X.unsqueeze(2)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_cont_cols': 4, 'embed_dim': 4}]
