import torch
from torch import nn


class Pooling(nn.Module):
    """Implementation of pooling for PoolFormer."""

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 
            2, count_include_pad=False)

    def forward(self, x):
        """Forward function."""
        return self.pool(x) - x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
