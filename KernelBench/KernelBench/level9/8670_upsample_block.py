import torch
import torch.nn as nn
import torch.nn.functional as F


class upsample_block(nn.Module):
    """
    Defines upsampling block. The upsampling is performed
    using bilinear or nearest interpolation followed by 1-by-1
    convolution (the latter can be used to reduce
    a number of feature channels).
    """

    def __init__(self, input_channels, output_channels, scale_factor=2,
        mode='bilinear'):
        """Initializes module parameters"""
        super(upsample_block, self).__init__()
        assert mode == 'bilinear' or mode == 'nearest', "use 'bilinear' or 'nearest' for upsampling mode"
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=
            1, stride=1, padding=0)
        self.upsample2x = nn.ConvTranspose2d(input_channels, input_channels,
            kernel_size=3, stride=2, padding=(1, 1), output_padding=(1, 1))

    def forward(self, x):
        """Defines a forward path"""
        if self.scale_factor == 2:
            x = self.upsample2x(x)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode
                )
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
