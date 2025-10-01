import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Matmul, BatchNorm, ReLU, Add, Max, and MaxPool operations.

    Operators used:
    - Matmul
    - BatchNorm
    - ReLU
    - Add
    - MaxPool
    """
    def __init__(self, num_features, num_channels):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.linear = nn.Linear(num_channels, num_channels)

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, num_channels).
            y (torch.Tensor): Input tensor of shape (batch_size, num_channels, num_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_channels/2).
        """
        x = torch.matmul(x, y)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.max(x, dim=2)[0]
        return x

def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 1024
    num_features = 64
    num_channels = 512
    y_channels = 128

    x = torch.randn(batch_size, num_features, num_channels)
    y = torch.randn(batch_size, num_channels, y_channels)
    return [x, y]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    num_features = 128
    num_channels = 128
    return [num_features, num_channels]