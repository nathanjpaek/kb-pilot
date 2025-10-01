import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that uses ConvTranspose2d, GELU, Add, and Scale.
    Operators used: ConvTranspose2d, GELU, Scale, Add, GELU, Add
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1)
        self.scale = nn.Parameter(torch.ones(1))
        self.add = nn.Parameter(torch.zeros(1))

    def forward(self, x, residual):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            residual (torch.Tensor): Residual tensor of shape (batch_size, out_channels, height*2, width*2).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height*2, width*2).
        """
        x = self.conv_transpose(x)
        x = torch.nn.functional.gelu(x)
        x = x * self.scale
        x = x + self.add
        x = torch.nn.functional.gelu(x)
        x = x + residual
        return x


def get_inputs():
    batch_size = 256
    in_channels = 32
    out_channels = 64
    height, width = 32, 32

    x = torch.randn(batch_size, in_channels, height, width)
    residual = torch.randn(batch_size, out_channels, height * 2, width * 2)
    return [x, residual]

def get_init_inputs():
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    scale_factor = 2
    return [in_channels, out_channels, kernel_size, scale_factor]