import torch
import torch.nn as nn


class LayerNorm2D(nn.Module):
    """Layer normalization for CNN outputs."""

    def __init__(self, channel, idim, eps=1e-12):
        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm([channel, idim], eps=eps)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, C, T, F]`
        Returns:
            xs (FloatTensor): `[B, C, T, F]`

        """
        _B, _C, _T, _F = xs.size()
        xs = xs.transpose(2, 1).contiguous()
        xs = self.norm(xs)
        xs = xs.transpose(2, 1)
        return xs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4, 'idim': 4}]
