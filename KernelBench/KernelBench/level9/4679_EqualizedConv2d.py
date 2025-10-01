import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.functional
from typing import List
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


class EqualizedConv2d(nn.Module):
    """
    <a id="equalized_conv2d"></a>
    ## Learning-rate Equalized 2D Convolution Layer

    This uses [learning-rate equalized weights]($equalized_weights) for a convolution layer.
    """

    def __init__(self, in_features: 'int', out_features: 'int', kernel_size:
        'int', padding: 'int'=0):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `padding` is the padding to be added on both sides of each size dimension
        """
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features,
            kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: 'torch.Tensor'):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'kernel_size': 4}]
