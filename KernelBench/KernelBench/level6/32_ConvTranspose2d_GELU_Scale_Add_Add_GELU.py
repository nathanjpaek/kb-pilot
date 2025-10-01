import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, adds a scaled tensor, adds another tensor, and applies GELU again.
    Operators used: ConvTranspose2d, GELU, Scale, Add, GELU, Add
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.scale = nn.Parameter(torch.tensor(scale_factor))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.add = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height*kernel_size, width*kernel_size).
        """
        x = self.conv_transpose(x)
        x = torch.nn.functional.gelu(x)
        x = x + self.scale * self.bias
        x = x + self.add
        x = torch.nn.functional.gelu(x)
        return x

def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 1024
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
    scale_factor = 0.5
    bias_shape = (1, out_channels, 1, 1)

    return [in_channels, out_channels, kernel_size, scale_factor, bias_shape]