import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs ConvTranspose3d, Scaling, Softmax, and InstanceNorm operations.
    Operators: ConvTranspose3d, Scaling, Softmax, InstanceNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth*scale_factor, height*scale_factor, width*scale_factor).
        """
        x = self.conv_transpose(x)
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        x = torch.softmax(x, dim=1)
        x = self.instance_norm(x)
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 64
    in_channels = 8
    depth, height, width = 16, 32, 32

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_channels = 8
    out_channels = 16
    kernel_size = 3
    scale_factor = 2

    return [in_channels, out_channels, kernel_size, scale_factor]