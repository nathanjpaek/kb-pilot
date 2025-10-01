import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, applies HardSwish activation, and then applies average pooling.
    Operators used: Conv2d, HardSwish, AvgPool
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.hardswish = nn.Hardswish()
        self.avgpool = nn.AvgPool2d(pool_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after convolution, HardSwish, and average pooling.
        """
        x = self.conv(x)
        x = self.hardswish(x)
        x = self.avgpool(x)
        return x


def get_inputs():
    """
    Returns a list containing a single input tensor for the model.
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
    pool_size = 2

    return [in_channels, out_channels, kernel_size, pool_size]