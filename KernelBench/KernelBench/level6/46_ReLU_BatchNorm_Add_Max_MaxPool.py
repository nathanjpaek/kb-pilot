import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Matmul, BatchNorm, ReLU, Add, Max, and MaxPool operations.
    Operators used: Matmul, BatchNorm, ReLU, Add, MaxPool
    """
    def __init__(self, num_features, num_channels, height, width, maxpool_kernel_size):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.linear = nn.Linear(num_features, num_channels * height * width)

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            y (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_channels, height//maxpool_kernel_size, width//maxpool_kernel_size).
        """
        x = torch.relu(self.linear(x))
        x = x.reshape(y.shape)
        x = self.bn(x)
        x = x + y
        x = torch.max(x, dim=1, keepdim=True)[0]
        x = self.maxpool(x)
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 256
    num_features = 128
    num_channels = 32
    height, width = 64, 64

    return [torch.randn(batch_size, num_features), torch.randn(batch_size, num_channels, height, width)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    num_features = 128
    num_channels = 32
    height, width = 64, 64
    maxpool_kernel_size = 2

    return [num_features, num_channels, height, width, maxpool_kernel_size]