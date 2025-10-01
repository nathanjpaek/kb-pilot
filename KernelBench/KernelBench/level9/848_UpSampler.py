import torch
from torch import nn
from torch.nn import functional as F


class UpSampler(nn.Module):
    """Up Sample module
    
    Decrease the channels size and increase the spatial size of tensor
    
    Extends:
        nn.Module
    """

    def __init__(self, inChannels, outChannels, spatial_size):
        """
        Arguments:
            inChannels {int} -- Number of in channels
            outChannels {int} -- Number of out channels
            spatial_size {tuple} -- Spatial size to get
        """
        super(UpSampler, self).__init__()
        self.spatial_size = spatial_size
        self.conv = nn.Conv3d(inChannels, outChannels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, self.spatial_size, mode='trilinear',
            align_corners=False)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inChannels': 4, 'outChannels': 4, 'spatial_size': 4}]
