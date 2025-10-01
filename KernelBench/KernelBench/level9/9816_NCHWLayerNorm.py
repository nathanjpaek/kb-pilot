import torch
from torch import nn


class NCHWLayerNorm(nn.LayerNorm):
    """Applies LayerNorm to the channel dimension of NCHW tensors."""

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'normalized_shape': 4}]
