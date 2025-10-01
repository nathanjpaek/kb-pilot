import torch
import torch as th
import torch.nn as nn
from torch.nn import Parameter


class BaLayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf

    This implementation mimicks the original torch implementation at:
    https://github.com/ryankiros/layer-norm/blob/master/torch_modules/LayerNormalization.lua
    """

    def __init__(self, input_size: 'int', learnable: 'bool'=True, epsilon:
        'float'=1e-05):
        super(BaLayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.epsilon = epsilon
        self.alpha = th.empty(1, input_size).fill_(0)
        self.beta = th.empty(1, input_size).fill_(0)
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)

    def forward(self, x: 'th.Tensor') ->th.Tensor:
        size = x.size()
        x = x.view(x.size(0), -1)
        mean = th.mean(x, 1).unsqueeze(1)
        center = x - mean
        std = th.sqrt(th.mean(th.square(center), 1)).unsqueeze(1)
        output = center / (std + self.epsilon)
        if self.learnable:
            output = self.alpha * output + self.beta
        return output.view(size)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
