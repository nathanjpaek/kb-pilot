import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Matmul, BatchNorm, ReLU, Add, Max, and MaxPool operations.
    Operators used: Matmul, BatchNorm, ReLU, Add, Max, MaxPool
    """
    def __init__(self, num_features, num_channels, matmul_dim):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.matmul_dim = matmul_dim
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.add_tensor = nn.Parameter(torch.randn(matmul_dim))
        self.num_channels = num_channels

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            y (torch.Tensor): Input tensor of shape (batch_size, num_features, matmul_dim).

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = torch.relu(x)
        x = self.bn(x)
        x = torch.matmul(x.unsqueeze(1), y).squeeze(1)
        x = x + self.add_tensor
        x = torch.max(x, dim=1, keepdim=True)[0]
        x = x.view(-1, self.num_channels, self.matmul_dim // self.num_channels)
        x = self.maxpool(x).squeeze()

        return x


def get_inputs():
    batch_size = 2048
    num_features = 512
    matmul_dim = 256
    num_channels = 8

    x = torch.randn(batch_size, num_features)
    y = torch.randn(batch_size, num_features, matmul_dim)
    return [x, y]

def get_init_inputs():
    num_features = 512
    num_channels = 8
    matmul_dim = 256
    return [num_features, num_channels, matmul_dim]