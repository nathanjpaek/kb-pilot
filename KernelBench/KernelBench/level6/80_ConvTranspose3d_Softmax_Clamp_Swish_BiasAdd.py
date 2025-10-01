import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model using ConvTranspose3d, Swish, Clamp, Softmax, and BiasAdd.

    Operators used: ConvTranspose3d, Swish, Clamp, Softmax, BiasAdd
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = self.conv_transpose(x)
        x = F.softmax(x, dim=2)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = x * torch.sigmoid(x) # Swish Activation
        x = x + self.bias
        return x

def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 128
    in_channels = 4
    depth, height, width = 16, 32, 32
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    stride = 2
    padding = 1
    bias_shape = (out_channels, 1, 1, 1)
    clamp_min = 0.1
    clamp_max = 0.9
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape, clamp_min, clamp_max]