import math
import torch
import numpy as np
from torch import nn
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


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'shape': [4, 4]}]
