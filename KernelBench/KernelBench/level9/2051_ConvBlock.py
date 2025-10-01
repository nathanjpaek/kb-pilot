import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(self, input_channels: 'int', output_channels: 'int',
        kernel_size: 'Param2D'=3, stride: 'Param2D'=1, padding: 'Param2D'=1
        ) ->None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=
            kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        c = self.conv(x)
        r = self.relu(c)
        return r


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
