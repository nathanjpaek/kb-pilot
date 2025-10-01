from torch.nn import Module
import torch
from torch import Tensor
from torch.nn.modules import Module
import torch.optim.lr_scheduler


class L2Normalization(Module):
    """Module to L2-normalize the input. Typically used in last layer to
    normalize the embedding."""

    def __init__(self):
        super().__init__()

    def forward(self, x: 'Tensor') ->Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
