import torch
import torch.nn as nn


class AdaptiveConcatPool2d(nn.Module):
    """
    Pools with AdaptiveMaxPool2d AND AdaptiveAvgPool2d and concatenates both
    results.

    Args:
        target_size: the target output size (single integer or
            double-integer tuple)
    """

    def __init__(self, target_size):
        super(AdaptiveConcatPool2d, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        return torch.cat([nn.functional.adaptive_avg_pool2d(x, self.
            target_size), nn.functional.adaptive_max_pool2d(x, self.
            target_size)], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'target_size': 4}]
