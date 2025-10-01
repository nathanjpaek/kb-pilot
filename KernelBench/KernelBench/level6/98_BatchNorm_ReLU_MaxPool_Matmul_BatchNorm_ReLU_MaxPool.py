import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model using Matmul, BatchNorm, ReLU, Add, Max, and MaxPool.
    Operators used: Matmul, BatchNorm, ReLU, Add, MaxPool, Max
    """
    def __init__(self, num_features, num_channels, height, width):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(num_channels * (height // 2) * (width // 2), num_features)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).
            y (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        y = self.bn(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = torch.flatten(y, 1)
        y = self.linear(y)
        
        # Reshape to enable matmul
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, num_features)
        y = y.unsqueeze(2)  # Shape: (batch_size, num_features, 1)
        
        out = torch.matmul(x, y)  # Shape: (batch_size, 1, 1)
        out = out.squeeze(1).squeeze(1) # Shape: (batch_size)
        return out


def get_inputs():
    batch_size = 256
    num_channels = 32
    height, width = 64, 64
    return [torch.randn(batch_size, num_channels, height, width), torch.randn(batch_size, num_channels, height, width)]

def get_init_inputs():
    num_features = 128
    num_channels = 32
    height, width = 64, 64
    return [num_features, num_channels, height, width]