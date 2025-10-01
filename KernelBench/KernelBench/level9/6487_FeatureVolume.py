import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureVolume(nn.Module):

    def __init__(self, fdim, fsize):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        var = 0.01
        self.fmx = nn.Parameter(torch.randn(1, fdim, fsize, fsize) * var)
        self.sparse = None
        self.padding_mode = 'reflection'

    def forward(self, x):
        N = x.shape[0]
        if len(x.shape) == 3:
            sample_coords = x.reshape(1, N, x.shape[1], 3)
            sampley = F.grid_sample(self.fmx, sample_coords[..., [0, 2]],
                align_corners=True, padding_mode=self.padding_mode)[0, :, :, :
                ].transpose(0, 1)
        else:
            sample_coords = x.reshape(1, N, 1, 3)
            sampley = F.grid_sample(self.fmx, sample_coords[..., [0, 2]],
                align_corners=True, padding_mode=self.padding_mode)[0, :, :, 0
                ].transpose(0, 1)
        return sampley


def get_inputs():
    return [torch.rand([1, 1, 1, 3])]


def get_init_inputs():
    return [[], {'fdim': 4, 'fsize': 4}]
