import torch
import torch.nn as nn


class Fusion(nn.Module):
    """ Base Fusion Class"""

    def __init__(self, input_dim=3):
        super().__init__()
        self.input_dim = input_dim

    def tile_x2(self, x1, x2, x2_proj=None):
        if x2_proj:
            x2 = x2_proj(x2)
        x2 = x2.unsqueeze(-1).unsqueeze(-1)
        x2 = x2.repeat(x1.shape[0], 1, x1.shape[-2], x1.shape[-1])
        return x2

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        raise NotImplementedError()


class FusionMax(Fusion):
    """ max(x1, x2) """

    def __init__(self, input_dim=3):
        super(FusionMax, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        return torch.max(x1, x2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
