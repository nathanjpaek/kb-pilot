import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class InterpolationBlock(nn.Module):
    """
    Interpolation block.

    Parameters:
    ----------
    scale_factor : float
        Multiplier for spatial size.
    """

    def __init__(self, scale_factor):
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(input=x, scale_factor=self.scale_factor, mode=
            'bilinear', align_corners=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale_factor': 1.0}]
