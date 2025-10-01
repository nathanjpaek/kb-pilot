import math
import torch
import torch as th
import torch.nn as nn
from torch.nn import Parameter


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size: 'int', learnable: 'bool'=True, epsilon:
        'float'=1e-06):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = th.empty(1, input_size).fill_(0)
        self.beta = th.empty(1, input_size).fill_(0)
        self.epsilon = epsilon
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x: 'th.Tensor') ->th.Tensor:
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - th.mean(x, 1).unsqueeze(1)) / th.sqrt(th.var(x, 1).
            unsqueeze(1) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
