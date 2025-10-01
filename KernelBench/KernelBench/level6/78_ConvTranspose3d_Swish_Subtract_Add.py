import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies Swish activation, subtracts a value, and adds another value.
    Operators used: ConvTranspose3d, Swish, Subtract, Add
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        self.swish = nn.SiLU()
        self.subtract_value = nn.Parameter(torch.randn(1))
        self.add_value = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        x = self.conv_transpose(x)
        x = self.swish(x)
        x = x - self.subtract_value
        x = x + self.add_value
        return x


def get_inputs():
    """
    Returns a list containing a single input tensor for the model.
    """
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 32, 32
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    stride = 2
    padding = 1
    return [in_channels, out_channels, kernel_size, stride, padding]