import torch
import torch.nn as nn


class ChanLayerNorm(nn.Module):
    """Channelwise LayerNorm"""

    def __init__(self, d: 'int', **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(d, **kwargs)

    def forward(self, x):
        x = self.ln(x.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2).contiguous()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d': 4}]
