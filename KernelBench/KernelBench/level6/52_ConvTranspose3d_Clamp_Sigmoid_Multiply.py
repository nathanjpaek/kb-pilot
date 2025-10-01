import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs ConvTranspose3d, Clamp, Sigmoid, and Multiply operations.
    Operators used: ConvTranspose3d, Clamp, Sigmoid, Multiply
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth', height', width').
        """
        x = self.conv_transpose(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = torch.sigmoid(x)
        x = x * 2.0
        return x


def get_inputs():
    """
    Returns a list containing a randomly generated input tensor for the model.
    """
    batch_size = 256
    in_channels = 3
    depth, height, width = 16, 32, 32

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list containing the initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 8
    kernel_size = 3

    return [in_channels, out_channels, kernel_size]