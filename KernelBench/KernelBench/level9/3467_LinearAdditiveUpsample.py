import torch
import torch.utils.data
import torch
import torch.nn as nn


class LinearAdditiveUpsample(nn.Module):
    """Bi/Trilinear Additive Upsample

    Upsampling strategy described in Wojna et al (https://doi.org/10.1007/s11263-019-01170-8) to avoid checkerboard
    patterns while keeping a better performance for the convolution.

    Parameters:
        scale_factor (int)  -- the factor for the upsampling operation
        n_splits (float)    -- the channel reduction factor
        threed (bool)       -- 3D (true) or 2D (false) network

    """

    def __init__(self, scale_factor, n_splits, threed):
        super(LinearAdditiveUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.n_splits = n_splits
        if threed:
            self.mode = 'trilinear'
        else:
            self.mode = 'bilinear'

    def forward(self, input_tensor):
        n_channels = input_tensor.shape[1]
        assert self.n_splits > 0 and n_channels % self.n_splits == 0, 'Number of feature channels should be divisible by n_splits'
        resizing_layer = nn.functional.interpolate(input_tensor,
            scale_factor=self.scale_factor, mode=self.mode, align_corners=False
            )
        split = torch.split(resizing_layer, self.n_splits, dim=1)
        split_tensor = torch.stack(split, dim=1)
        output_tensor = torch.sum(split_tensor, dim=2)
        return output_tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale_factor': 1.0, 'n_splits': 4, 'threed': 4}]
