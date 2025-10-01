import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs ConvTranspose2d, Mean, LayerNorm, Hardtanh, LayerNorm.
    Operators used: ConvTranspose2d, Mean, LayerNorm, Hardtanh, LayerNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.layer_norm1 = nn.LayerNorm(norm_shape)
        self.hardtanh = nn.Hardtanh()
        self.layer_norm2 = nn.LayerNorm(norm_shape)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv_transpose(x)
        x = torch.mean(x, dim=[2, 3])
        x = self.layer_norm1(x)
        x = self.hardtanh(x)
        x = self.layer_norm2(x)
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
    stride = 2
    norm_shape = out_channels
    return [in_channels, out_channels, kernel_size, stride, norm_shape]