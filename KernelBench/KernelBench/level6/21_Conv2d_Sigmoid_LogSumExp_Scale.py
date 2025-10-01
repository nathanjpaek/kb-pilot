import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, applies Sigmoid, LogSumExp, and Scale.
    Operators: Conv2d, Sigmoid, LogSumExp, Scale
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * self.scale_factor
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 256
    in_channels = 3
    height, width = 128, 128
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    scale_factor = 2.0
    return [in_channels, out_channels, kernel_size, scale_factor]