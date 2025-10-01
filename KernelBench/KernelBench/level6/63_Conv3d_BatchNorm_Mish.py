import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies batch normalization, and then the Mish activation function.
    Operators: Conv3d, BatchNorm, Mish
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm3d(out_channels)
        self.mish = nn.Mish()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth', height', width').
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x

def get_inputs():
    """
    Returns a list containing a single randomly generated input tensor for the model.
    """
    batch_size = 256
    in_channels = 3
    depth, height, width = 32, 16, 16

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list containing the initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 16
    kernel_size = 3

    return [in_channels, out_channels, kernel_size]