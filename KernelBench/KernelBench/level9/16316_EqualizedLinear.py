import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from typing import List
import torch.nn.functional
import torch.autograd


class EqualizedWeight(nn.Module):
    """
    <a id="equalized_weight"></a>

    ## Learning-rate Equalized Weights Parameter

    This is based on equalized learning rate introduced in the Progressive GAN paper.
    Instead of initializing weights at $\\mathcal{N}(0,c)$ they initialize weights
    to $\\mathcal{N}(0, 1)$ and then multiply them by $c$ when using it.
    $$w_i = c \\hat{w}_i$$

    The gradients on stored parameters $\\hat{w}$ get multiplied by $c$ but this doesn't have
    an affect since optimizers such as Adam normalize them by a running mean of the squared gradients.

    The optimizer updates on $\\hat{w}$ are proportionate to the learning rate $\\lambda$.
    But the effective weights $w$ get updated proportionately to $c \\lambda$.
    Without equalized learning rate, the effective weights will get updated proportionately to just $\\lambda$.

    So we are effectively scaling the learning rate by $c$ for these weight parameters.
    """

    def __init__(self, shape: 'List[int]'):
        """
        * `shape` is the shape of the weight parameter
        """
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    """
    <a id="equalized_linear"></a>

    ## Learning-rate Equalized Linear Layer

    This uses [learning-rate equalized weights](#equalized_weights) for a linear layer.
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias:
        'float'=0.0):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: 'torch.Tensor'):
        return F.linear(x, self.weight(), bias=self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
