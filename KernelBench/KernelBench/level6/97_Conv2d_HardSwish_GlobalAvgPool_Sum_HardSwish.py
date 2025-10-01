# mean_runtime: 0.913 ms
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs Conv2d, HardSwish, GlobalAvgPool, Sum, and HardSwish operations.

    Operators: ['Conv2d', 'HardSwish', 'GlobalAvgPool', 'Sum', 'HardSwish']
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying the specified operations.
        """
        x = self.conv(x)
        x = F.hardswish(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.sum(x, dim=[2, 3])
        x = F.hardswish(x)
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
    Returns a list of arguments for initializing the model.
    """
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]