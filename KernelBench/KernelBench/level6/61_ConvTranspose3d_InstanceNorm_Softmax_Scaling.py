import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model combining ConvTranspose3d, Scaling, Softmax, and InstanceNorm.
    Operators used: ConvTranspose3d, InstanceNorm, Softmax, Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, instance_norm_features):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm3d(instance_norm_features)
        self.scaling = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth*kernel_size, height*kernel_size, width*kernel_size).
        """
        x = self.conv_transpose(x)
        x = self.instance_norm(x)
        x = torch.softmax(x * self.scaling, dim=1)
        return x


def get_inputs():
    batch_size = 256
    in_channels = 3
    depth, height, width = 16, 32, 32

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    instance_norm_features = out_channels

    return [in_channels, out_channels, kernel_size, instance_norm_features]