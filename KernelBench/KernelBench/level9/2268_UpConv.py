import torch
import torch.nn as nn
from enum import Enum
from enum import auto


class UpsampleType(Enum):
    CONV_TRANSPOSE = auto()
    NEAREST_NEIGHBOUR = auto()
    BILINEAR = auto()


class UpConv(nn.Module):
    """
    Custom module to handle a single Upsample + Convolution block used in the decoder layer.
    Takes an optional argument stating which type of upsampling to use. This argument should be provided from the
    UpsanmpleType enum above.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int',
        upsample_type: 'UpsampleType'=UpsampleType.CONV_TRANSPOSE, name=''):
        super().__init__()
        self.upsample = self._upsample(upsample_type, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            padding='same')
        self.name = name

    def _upsample(self, upsample_type: 'UpsampleType', num_channels: 'int'):
        if upsample_type == UpsampleType.CONV_TRANSPOSE:
            return nn.ConvTranspose2d(num_channels, num_channels,
                kernel_size=2, stride=2)
        if upsample_type == UpsampleType.NEAREST_NEIGHBOUR:
            return nn.UpsamplingNearest2d(scale_factor=2)
        if upsample_type == UpsampleType.BILINEAR:
            return nn.UpsamplingBilinear2d(scale_factor=2)
        raise NotImplementedError(
            f'Upsampling mode of {str(upsample_type)} is not supported.')

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.upsample(x)
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
