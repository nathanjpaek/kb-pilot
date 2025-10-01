import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Matmul, BatchNorm, ReLU, Add, Max, and MaxPool operations.
    Operators used: Matmul, BatchNorm, ReLU, Add, MaxPool
    """
    def __init__(self, num_features, num_channels):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.linear = nn.Linear(num_channels, num_channels)

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, num_channels).
            y (torch.Tensor): Input tensor of shape (batch_size, num_channels, num_channels).

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = torch.relu(x)
        x = torch.matmul(x, y)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.linear(x)
        x = self.maxpool(x)
        return x


def get_inputs():
    batch_size = 256
    num_features = 128
    num_channels = 64

    x = torch.randn(batch_size, num_features, num_channels)
    y = torch.randn(batch_size, num_channels, num_channels)
    return [x, y]

def get_init_inputs():
    num_features = 128
    num_channels = 64
    return [num_features, num_channels]