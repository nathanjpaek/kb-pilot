import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Matmul, BatchNorm, ReLU, Add, Max, and MaxPool operations.
    Operators: ['Matmul', 'BatchNorm', 'ReLU', 'Add', 'Max', 'MaxPool']
    """
    def __init__(self, num_features, num_channels, height, width, matmul_dim):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.matmul_dim = matmul_dim
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(matmul_dim, matmul_dim)

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).
            y (torch.Tensor): Input tensor of shape (batch_size, matmul_dim).

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        batch_size, num_channels, height, width = x.shape
        y = self.linear(y)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        x = x.view(batch_size, -1)
        x = torch.matmul(x, torch.randn(x.shape[1], self.matmul_dim).to(x.device))
        x = x + y
        x = torch.max(x, dim=1, keepdim=True)[0]
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 16
    num_channels = 32
    height, width = 64, 64
    matmul_dim = 128

    x = torch.randn(batch_size, num_channels, height, width)
    y = torch.randn(batch_size, matmul_dim)
    return [x, y]


def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    num_features = 128
    num_channels = 32
    height, width = 64, 64
    matmul_dim = 128
    return [num_features, num_channels, height, width, matmul_dim]