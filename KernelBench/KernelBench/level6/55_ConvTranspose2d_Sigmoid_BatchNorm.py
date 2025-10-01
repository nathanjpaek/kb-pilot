import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model with ConvTranspose2d, Sigmoid, and BatchNorm.

    Operators used:
    ConvTranspose2d, Sigmoid, BatchNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)

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
        x = self.bn(x)
        return x


def get_inputs():
    """
    Returns a list containing a randomly generated input tensor for the model.
    """
    batch_size = 256
    in_channels = 3
    height, width = 32, 32

    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    """
    Returns a list containing the initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 16
    kernel_size = 4

    return [in_channels, out_channels, kernel_size]