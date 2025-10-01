import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model with ConvTranspose2d, Swish, BiasAdd, Swish, Sigmoid layers.
    Operators: ConvTranspose2d, Swish, BiasAdd, Swish, Sigmoid
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height*kernel_size, width*kernel_size).
        """
        x = self.conv_transpose(x)
        x = torch.sigmoid(x)
        x = x + self.bias
        x = torch.sigmoid(x)
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 256
    in_channels = 32
    height, width = 32, 32

    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    bias_shape = (out_channels, 1, 1)

    return [in_channels, out_channels, kernel_size, bias_shape]