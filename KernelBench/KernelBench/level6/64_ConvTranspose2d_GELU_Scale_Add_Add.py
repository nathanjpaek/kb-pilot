import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model using ConvTranspose2d, GELU, Add, and Scale operations.
    Operators used: ConvTranspose2d, GELU, Scale, Add, Add
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
        x = x * self.scale + self.add
        x = x + residual
        x = x + residual
        return x


def get_inputs():
    batch_size = 256
    in_channels = 32
    out_channels = 16
    height, width = 32, 32
    residual_height, residual_width = height * 2, width * 2

    x = torch.randn(batch_size, in_channels, height, width)
    residual = torch.randn(batch_size, out_channels, residual_height, residual_width)

    return [x, residual]

def get_init_inputs():
    in_channels = 32
    out_channels = 16
    kernel_size = 3
    scale_factor = 2

    return [in_channels, out_channels, kernel_size, scale_factor]