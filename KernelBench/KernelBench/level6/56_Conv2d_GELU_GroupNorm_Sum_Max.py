import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model combining Conv2d, GELU, Sum, GroupNorm, and Max.
    Operators used: Conv2d, GELU, GroupNorm, Sum, Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.gn = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        residual = x
        x = torch.nn.functional.gelu(x)
        x = self.gn(x)
        x = x + residual
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.max(x, dim=3, keepdim=True)[0]
        return x


def get_inputs():
    """
    Returns a list containing a randomly generated input tensor.
    """
    batch_size = 256
    in_channels = 32
    height, width = 128, 128

    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    """
    Returns a list of arguments for initializing the model.
    """
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    num_groups = 8

    return [in_channels, out_channels, kernel_size, num_groups]