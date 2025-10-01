import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, subtracts the mean, and performs another subtraction.
    Operators used: Conv3d, Subtract, Mean
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth-kernel_size+1, height-kernel_size+1, width-kernel_size+1).
        """
        x = self.conv(x)
        mean = torch.mean(x, dim=[2, 3, 4], keepdim=True)
        x = x - mean
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 256
    in_channels = 3
    depth, height, width = 32, 32, 32

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 8
    kernel_size = 3

    return [in_channels, out_channels, kernel_size]