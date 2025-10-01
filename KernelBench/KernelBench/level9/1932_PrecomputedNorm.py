import torch
import torch.nn as nn


class PrecomputedNorm(nn.Module):
    """Normalization using Pre-computed Mean/Std.
    Args:
        stats: Precomputed (mean, std).
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, stats, axis=[1, 2]):
        super().__init__()
        self.axis = axis
        self.mean, self.std = stats

    def forward(self, X: 'torch.Tensor') ->torch.Tensor:
        return (X - self.mean) / self.std

    def __repr__(self):
        format_string = (self.__class__.__name__ +
            f'(mean={self.mean}, std={self.std}, axis={self.axis})')
        return format_string


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'stats': [4, 4]}]
