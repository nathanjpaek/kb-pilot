import math
import torch
from torch import Tensor
import torch.nn as nn


class BaseEmbeddingLayer(nn.Module):

    def _apply_initialization(self, x: 'Tensor', d: 'int', method: 'str'
        ) ->None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if method == 'uniform':
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif method == 'normal':
            nn.init.normal_(x, std=d_sqrt_inv)
        else:
            raise ValueError(f'initialization: {method} is not implemented')


class AppendCLSToken(BaseEmbeddingLayer):

    def __init__(self, d_token: 'int', initialization: 'str'='uniform') ->None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(d_token))
        self._apply_initialization(self.weight, d_token, initialization)

    def forward(self, x: 'Tensor') ->Tensor:
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1
            )], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_token': 4}]
