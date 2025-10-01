import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and batch normalization.
    Operators used: Conv3d, Mish, BatchNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth', height', width').
        """
        x = self.conv(x)
        x = nn.Mish()(x)
        x = self.bn(x)
        return x


def get_inputs():
    """
    Returns a list containing a single input tensor.

    The input tensor has shape (batch_size, in_channels, depth, height, width).
    """
    batch_size = 64
    in_channels = 3
    depth, height, width = 32, 64, 64

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.

    The arguments are in_channels, out_channels, and kernel_size.
    """
    in_channels = 3
    out_channels = 16
    kernel_size = 3

    return [in_channels, out_channels, kernel_size]