import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, scaling, and batch normalization.

    Operators Used:
    - Conv3d
    - Scaling
    - BatchNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_features):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm3d(num_features)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth', height', width').
        """
        x = self.conv(x)
        x = x * self.scaling
        x = self.bn(x)
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
    out_channels = 16
    kernel_size = 3
    num_features = out_channels

    return [in_channels, out_channels, kernel_size, num_features]