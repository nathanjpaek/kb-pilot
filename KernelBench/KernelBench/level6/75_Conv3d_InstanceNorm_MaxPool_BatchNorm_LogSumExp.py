import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs 3D convolution, InstanceNorm, MaxPool, BatchNorm, and LogSumExp.
    Operators: Conv3d, InstanceNorm, MaxPool, BatchNorm, LogSumExp
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel, num_features):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.instancenorm = nn.InstanceNorm3d(out_channels)
        self.maxpool = nn.MaxPool3d(pool_kernel)
        self.bn = nn.BatchNorm3d(out_channels)
        self.num_features = num_features


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor after applying the layers.
        """
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.maxpool(x)
        x = self.bn(x)
        x = torch.logsumexp(x, dim=[2, 3, 4])
        return x

def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 64
    in_channels = 3
    depth, height, width = 32, 64, 64

    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    pool_kernel = 2
    num_features = 8

    return [in_channels, out_channels, kernel_size, pool_kernel, num_features]