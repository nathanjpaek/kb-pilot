import torch
from typing import Sequence
import torch.nn as nn
import torch._C
import torch.serialization
import torch.nn.parallel


class AddPositionEmb(nn.Module):
    """Module to add position embedding to input features
    """

    def __init__(self, dim=384, spatial_shape=[14, 14]):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence
            ), f'"spatial_shape" must by a sequence or int, get {type(spatial_shape)} instead.'
        if len(spatial_shape) == 1:
            embed_shape = list(spatial_shape) + [dim]
        else:
            embed_shape = [dim] + list(spatial_shape)
        self.pos_embed = nn.Parameter(torch.zeros(1, *embed_shape))

    def forward(self, x):
        return x + self.pos_embed


def get_inputs():
    return [torch.rand([4, 384, 14, 14])]


def get_init_inputs():
    return [[], {}]
