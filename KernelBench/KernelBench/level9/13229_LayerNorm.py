import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        """Layer Norm."""
        super(LayerNorm, self).__init__(normalized_shape, eps=eps,
            elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = super(LayerNorm, self).forward(x)
        y = y.permute(0, 2, 1)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'normalized_shape': 4}]
