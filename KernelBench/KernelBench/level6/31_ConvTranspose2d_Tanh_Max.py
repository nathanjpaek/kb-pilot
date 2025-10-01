import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies Tanh activation, and then max.
    Operators: ConvTranspose2d, Tanh, Max
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying transposed convolution, Tanh activation, and max.
        """
        x = self.conv_transpose(x)
        x = torch.tanh(x)
        x = torch.max(x)
        return x

def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 256
    in_channels = 3
    height, width = 64, 64

    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 16
    kernel_size = 3

    return [in_channels, out_channels, kernel_size]