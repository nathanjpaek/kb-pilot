import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, subtracts the mean, and applies another 3D convolution.
    Operators used: Conv3d, Subtract, Mean
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth', height', width').
        """
        x = self.conv1(x)
        mean = torch.mean(x, dim=[2, 3, 4], keepdim=True)
        x = x - mean
        x = self.conv2(x)
        return x


def get_inputs():
    """
    Returns a list containing a single input tensor for the model.
    """
    batch_size = 32
    in_channels = 3
    depth, height, width = 32, 64, 64

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list of arguments for initializing the model.
    """
    in_channels = 3
    out_channels = 8
    kernel_size = 3

    return [in_channels, out_channels, kernel_size]