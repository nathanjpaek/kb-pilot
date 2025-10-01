import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model using Matmul, BatchNorm, ReLU, Add, Max, and MaxPool.
    Operators Used: [Matmul, BatchNorm, ReLU, Add, MaxPool, Max]
    """
    def __init__(self, num_features, num_channels, height, width):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(num_channels * (height // 2) * (width // 2), num_features)
        self.num_features = num_features
        self.num_channels = num_channels
        self.height = height
        self.width = width

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).
            y (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_features).
        """
        x = self.bn(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, self.num_channels * (self.height // 2) * (self.width // 2))
        x = self.linear(x)
        x = torch.matmul(x, y.transpose(0, 1))
        x = x.max()
        return x


def get_inputs():
    batch_size = 256
    num_channels = 32
    height, width = 64, 64
    num_features = 128

    x = torch.randn(batch_size, num_channels, height, width)
    y = torch.randn(batch_size, num_features)
    return [x, y]

def get_init_inputs():
    num_features = 128
    num_channels = 32
    height, width = 64, 64
    return [num_features, num_channels, height, width]