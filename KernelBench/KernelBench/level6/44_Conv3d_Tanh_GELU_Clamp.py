import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies GELU, Tanh, and Clamp.
    Operators: Conv3d, GELU, Tanh, Clamp
    """
    def __init__(self, in_channels, out_channels, kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth', height', width').
        """
        x = self.conv(x)
        x = torch.tanh(x)
        x = torch.nn.functional.gelu(x)
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 64
    in_channels = 3
    depth, height, width = 32, 64, 64

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    clamp_min = -0.5
    clamp_max = 0.5

    return [in_channels, out_channels, kernel_size, clamp_min, clamp_max]