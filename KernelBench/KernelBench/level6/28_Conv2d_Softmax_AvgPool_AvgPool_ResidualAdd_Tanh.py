import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Conv2d, Softmax, AvgPool, AvgPool, ResidualAdd, and Tanh operations.
    Operators used: Conv2d, Softmax, AvgPool, AvgPool, ResidualAdd, Tanh
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, in_features):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avgpool1 = nn.AvgPool2d(pool_size)
        self.avgpool2 = nn.AvgPool2d(pool_size)
        self.linear = nn.Linear(in_features, out_channels)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width').
        """
        residual = x
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        x = self.avgpool1(x)
        x = self.avgpool2(x)
        residual = self.linear(residual.flatten(1)).unsqueeze(-1).unsqueeze(-1)
        x = x + residual
        x = torch.tanh(x)
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
    pool_size = 2
    in_features = 3 * 128 * 128
    return [in_channels, out_channels, kernel_size, pool_size, in_features]