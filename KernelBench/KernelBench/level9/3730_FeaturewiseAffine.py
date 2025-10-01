import torch
from typing import Union
import torch.nn as nn


class FeaturewiseAffine(nn.Module):
    """Feature-wise affine layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x, scale: 'Union[float, torch.Tensor]', shift:
        'Union[float, torch.Tensor]'):
        res = scale * x + shift
        return res


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
