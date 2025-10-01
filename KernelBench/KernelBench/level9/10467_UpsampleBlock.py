import torch
import torch.nn.functional as F
import torch.nn as nn


class UpsampleBlock(nn.Module):
    """
    Defines upsampling block performed using bilinear
    or nearest-neigbor interpolation followed by 1-by-1 convolution
    (the latter can be used to reduce a number of feature channels)

    Args:
        ndim:
            Data dimensionality (1D or 2D)
        input_channels:
            Number of input channels for the block
        output_channels:
            Number of the output channels for the block
        scale_factor:
            Scale factor for upsampling
        mode:
            Upsampling mode. Select between "bilinear" and "nearest"
    """

    def __init__(self, ndim: 'int', input_channels: 'int', output_channels:
        'int', scale_factor: 'int'=2, mode: 'str'='bilinear') ->None:
        """
        Initializes module parameters
        """
        super(UpsampleBlock, self).__init__()
        if not any([mode == 'bilinear', mode == 'nearest']):
            raise NotImplementedError(
                "use 'bilinear' or 'nearest' for upsampling mode")
        if not 0 < ndim < 3:
            raise AssertionError('ndim must be equal to 1 or 2')
        conv = nn.Conv2d if ndim == 2 else nn.Conv1d
        self.scale_factor = scale_factor
        self.mode = mode if ndim == 2 else 'nearest'
        self.conv = conv(input_channels, output_channels, kernel_size=1,
            stride=1, padding=0)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Defines a forward pass
        """
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'ndim': 1, 'input_channels': 4, 'output_channels': 4}]
