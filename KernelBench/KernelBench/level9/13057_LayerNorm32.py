import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):

    def forward(self, x):
        return super().forward(x.float().transpose(1, 2)).type(x.dtype
            ).transpose(1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'normalized_shape': 4}]
