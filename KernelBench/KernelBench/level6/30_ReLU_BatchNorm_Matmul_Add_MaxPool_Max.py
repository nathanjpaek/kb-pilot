import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Matmul, BatchNorm, ReLU, Add, Max, and MaxPool operations.
    Operators used: Matmul, BatchNorm, ReLU, Add, Max, MaxPool
    """
    def __init__(self, num_features, num_channels, height, width, matmul_dim):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.matmul_dim = matmul_dim
        self.num_channels = num_channels
        self.height = height
        self.width = width

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            y (torch.Tensor): Input tensor of shape (batch_size, matmul_dim).

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = torch.relu(x)
        x = self.bn(x)
        x = torch.matmul(x, torch.randn(x.shape[-1], self.matmul_dim).to(x.device))
        x = x + y
        x = x.view(-1, self.num_channels, self.height, self.width)
        x = self.maxpool(x)
        x = torch.max(x)
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 2048
    num_features = 512
    matmul_dim = 256
    num_channels = 16
    height, width = 32, 32

    x = torch.randn(batch_size, num_features)
    y = torch.randn(batch_size, matmul_dim)
    return [x, y]


def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    num_features = 512
    num_channels = 16
    height, width = 32, 32
    matmul_dim = 256
    return [num_features, num_channels, height, width, matmul_dim]