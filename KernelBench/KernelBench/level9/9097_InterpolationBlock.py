import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data.distributed


class InterpolationBlock(nn.Module):
    """
    Interpolation upsampling block.

    Parameters:
    ----------
    scale_factor : float
        Multiplier for spatial size.
    mode : str, default 'bilinear'
        Algorithm used for upsampling.
    align_corners : bool, default True
        Whether to align the corner pixels of the input and output tensors.
    """

    def __init__(self, scale_factor, mode='bilinear', align_corners=True):
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(input=x, scale_factor=self.scale_factor, mode=
            self.mode, align_corners=self.align_corners)

    def __repr__(self):
        s = (
            '{name}(scale_factor={scale_factor}, mode={mode}, align_corners={align_corners})'
            )
        return s.format(name=self.__class__.__name__, scale_factor=self.
            scale_factor, mode=self.mode, align_corners=self.align_corners)

    def calc_flops(self, x):
        assert x.shape[0] == 1
        if self.mode == 'bilinear':
            num_flops = 9 * x.numel()
        else:
            num_flops = 4 * x.numel()
        num_macs = 0
        return num_flops, num_macs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale_factor': 1.0}]
