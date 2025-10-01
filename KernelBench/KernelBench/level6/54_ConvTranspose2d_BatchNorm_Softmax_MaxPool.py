import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs ConvTranspose2d, BatchNorm, Softmax, and MaxPool.
    Operators used: ConvTranspose2d, BatchNorm, Softmax, MaxPool
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm_features, pool_kernel_size, pool_stride):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(batchnorm_features)
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying the layers.
        """
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = torch.softmax(x, dim=1)
        x = self.maxpool(x)
        return x


def get_inputs():
    batch_size = 256
    in_channels = 32
    height, width = 32, 32

    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    in_channels = 32
    out_channels = 64
    kernel_size = 4
    stride = 2
    padding = 1
    batchnorm_features = out_channels
    pool_kernel_size = 2
    pool_stride = 2

    return [in_channels, out_channels, kernel_size, stride, padding, batchnorm_features, pool_kernel_size, pool_stride]