import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs ConvTranspose2d, GELU, Add, Scale, and Add operations.
    Operators used: ConvTranspose2d, GELU, Add, Scale, Add
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=scale_factor, padding=1, output_padding=1)
        self.gelu = nn.GELU()
        self.scale = nn.Parameter(torch.ones(1))
        self.add_tensor = nn.Parameter(torch.randn(1, out_channels, 1, 1))

    def forward(self, x, residual):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            residual (torch.Tensor): Residual tensor to be added of shape (batch_size, out_channels, height*scale_factor, width*scale_factor).

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = self.conv_transpose(x)
        x = self.gelu(x)
        x = x + self.add_tensor
        x = x * self.scale
        x = x + residual
        return x

def get_inputs():
    batch_size = 256
    in_channels = 32
    out_channels = 64
    height, width = 32, 32
    scale_factor = 2
    
    x = torch.randn(batch_size, in_channels, height, width)
    residual = torch.randn(batch_size, out_channels, height * scale_factor, width * scale_factor)

    return [x, residual]

def get_init_inputs():
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    scale_factor = 2
    return [in_channels, out_channels, kernel_size, scale_factor]