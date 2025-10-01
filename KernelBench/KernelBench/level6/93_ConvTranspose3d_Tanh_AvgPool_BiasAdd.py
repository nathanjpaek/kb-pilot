import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies Tanh activation, 
    performs average pooling, and adds a bias term.
    Operators: ConvTranspose3d, Tanh, AvgPool, BiasAdd
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape, pool_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_size, stride=pool_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth', height', width').
        """
        x = self.conv_transpose(x)
        x = torch.tanh(x)
        x = self.avg_pool(x)
        x = x + self.bias
        return x


def get_inputs():
    """
    Returns a list containing a randomly generated input tensor for the model.
    """
    batch_size = 64
    in_channels = 8
    depth, height, width = 16, 32, 32

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list of arguments for initializing the model.
    """
    in_channels = 8
    out_channels = 16
    kernel_size = 3
    stride = 2
    padding = 1
    bias_shape = (out_channels, 1, 1, 1)
    pool_size = 2

    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape, pool_size]