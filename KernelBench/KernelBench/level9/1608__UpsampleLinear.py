import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class _UpsampleLinear(nn.Module):

    def __init__(self, scale):
        super(_UpsampleLinear, self).__init__()
        self._mode = 'linear', 'bilinear', 'trilinear'
        self.scale = scale

    def forward(self, x, scale=None):
        scale = scale or self.scale
        mode = self._mode[x.dim() - 3]
        return F.interpolate(x, scale_factor=scale, mode=mode,
            align_corners=False)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
