import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, finds the minimum and maximum values along the channel dimension.
    Operators used: Conv3d, Min, Max
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
            torch.Tensor: Output tensor of shape (batch_size, 1, depth, height, width).
        """
        x = self.conv(x)
        min_vals = torch.min(x, dim=1, keepdim=True)[0]
        max_vals = torch.max(x, dim=1, keepdim=True)[0]
        x = min_vals + max_vals
        return x

def get_inputs():
    """
    Returns a list containing a single input tensor for the model.
    """
    batch_size = 256
    in_channels = 3
    depth, height, width = 32, 32, 32

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list containing the initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 16
    kernel_size = 3

    return [in_channels, out_channels, kernel_size]