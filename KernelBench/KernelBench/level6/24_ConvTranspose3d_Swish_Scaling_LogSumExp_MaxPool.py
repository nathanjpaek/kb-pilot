import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs ConvTranspose3d, Swish, Scaling, LogSumExp, and MaxPool operations.
    Operators used: ConvTranspose3d, Swish, Scaling, LogSumExp, MaxPool
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        self.scaling = nn.Parameter(torch.tensor(scale_factor))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = self.conv_transpose(x)
        x = x * torch.sigmoid(x) * self.scaling  # Swish activation
        x = torch.logsumexp(x, dim=2)  # LogSumExp along the depth dimension
        x = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))(x)

        return x


def get_inputs():
    """
    Returns a list containing a randomly generated input tensor for the model.
    """
    batch_size = 128
    in_channels = 8
    depth, height, width = 32, 64, 64

    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    """
    Returns a list containing the initialization arguments for the model.
    """
    in_channels = 8
    out_channels = 16
    kernel_size = 3
    stride = 1
    padding = 1
    scale_factor = 2.0

    return [in_channels, out_channels, kernel_size, stride, padding, scale_factor]