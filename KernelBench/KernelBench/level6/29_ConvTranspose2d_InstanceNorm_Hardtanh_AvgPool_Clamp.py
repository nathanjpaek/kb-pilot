import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model using ConvTranspose2d, Hardtanh, InstanceNorm, Clamp, Min, AvgPool.
    Operators used: ConvTranspose2d, InstanceNorm, Hardtanh, AvgPool, Clamp
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, instance_norm_features, min_val, max_val):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.instance_norm = nn.InstanceNorm2d(instance_norm_features)
        self.hardtanh = nn.Hardtanh()
        self.min_val = min_val
        self.max_val = max_val
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height/2, width/2).
        """
        x = self.conv_transpose(x)
        x = self.instance_norm(x)
        x = self.hardtanh(x)
        x = self.avgpool(x)
        x = torch.clamp(x, self.min_val, self.max_val)
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
    kernel_size = 4
    stride = 2
    padding = 1
    instance_norm_features = out_channels
    min_val = -0.5
    max_val = 0.5

    return [in_channels, out_channels, kernel_size, stride, padding, instance_norm_features, min_val, max_val]