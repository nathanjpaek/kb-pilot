import math
import torch
import typing
from torch import nn
from torch.nn import LayerNorm


def gelu(x: 'torch.Tensor') ->torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x: 'torch.Tensor') ->torch.Tensor:
    return x * torch.sigmoid(x)


def get_activation_fn(name: 'str') ->typing.Callable:
    if name == 'gelu':
        return gelu
    elif name == 'relu':
        return nn.functional.relu
    elif name == 'swish':
        return swish
    else:
        raise ValueError(f'Unrecognized activation fn: {name}')


class PredictionHeadTransform(nn.Module):

    def __init__(self, hidden_size: 'int', input_size: 'int', hidden_act:
        'typing.Union[str, typing.Callable]'='gelu', layer_norm_eps:
        'float'=1e-12):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = get_activation_fn(hidden_act)
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'input_size': 4}]
