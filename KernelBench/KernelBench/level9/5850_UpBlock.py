import torch
from torch import nn
import torch.nn.functional as F


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=
        padding, bias=bias, stride=stride)


class UpBlock(nn.Module):
    """
        A module down sample the feature map
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the convolving kernel
            order (string): determines the order of layers, e.g.
                'cr' -> conv + ReLU
                'crg' -> conv + ReLU + groupnorm
            num_groups (int): number of groups for the GroupNorm
        """

    def __init__(self, input_channels, output_channels):
        super(UpBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = conv3d(input_channels, output_channels, kernel_size=1,
            bias=True, padding=0)

    def forward(self, x):
        _, _c, w, h, d = x.size()
        upsample1 = F.upsample(x, [2 * w, 2 * h, 2 * d], mode='trilinear')
        upsample = self.conv1(upsample1)
        return upsample


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
