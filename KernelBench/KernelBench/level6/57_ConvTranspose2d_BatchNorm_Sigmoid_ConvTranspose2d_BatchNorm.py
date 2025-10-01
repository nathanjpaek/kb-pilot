import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs ConvTranspose2d, Sigmoid, and BatchNorm operations.
    Operators used: ConvTranspose2d, BatchNorm, Sigmoid, ConvTranspose2d, BatchNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv_transpose2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying the layers.
        """
        x = self.conv_transpose1(x)
        x = self.batch_norm1(x)
        x = torch.sigmoid(x)
        x = self.conv_transpose2(x)
        x = self.batch_norm2(x)
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
    return [in_channels, out_channels, kernel_size]