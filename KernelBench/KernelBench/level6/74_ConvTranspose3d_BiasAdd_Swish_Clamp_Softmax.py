import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model using ConvTranspose3d, Swish, Clamp, Softmax, and BiasAdd.
    Operators: [ConvTranspose3d, Swish, Clamp, Softmax, BiasAdd]
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor after applying ConvTranspose3d, Swish, Clamp, Softmax, and BiasAdd.
        """
        x = self.conv_transpose(x)
        x = x + self.bias
        x = F.silu(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = F.softmax(x, dim=1)
        return x

def get_inputs():
    """
    Returns a list containing a randomly generated input tensor for the model.
    """
    batch_size = 128
    in_channels = 8
    depth, height, width = 32, 32, 32

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list containing the initialization arguments for the model.
    """
    in_channels = 8
    out_channels = 16
    kernel_size = 3
    bias_shape = (1, out_channels, 1, 1, 1)

    return [in_channels, out_channels, kernel_size, bias_shape]