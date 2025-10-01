import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model combining Matmul, BatchNorm, ReLU, Add, Max, and MaxPool.
    Operators used: Matmul, BatchNorm, ReLU, Add, MaxPool, Max
    """
    def __init__(self, num_features, num_channels):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            y (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.matmul(x, y.transpose(0, 1))
        x = self.bn(x)
        x = torch.relu(x)
        x = x + 1
        x = self.maxpool(x)
        x = torch.max(x, dim=1, keepdim=True)[0]
        return x


def get_inputs():
    """
    Returns a list of random input tensors for the model.
    """
    batch_size = 2048
    num_features = 1024

    return [torch.randn(batch_size, num_features), torch.randn(batch_size, num_features)]


def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    num_features = 2048
    num_channels = 64

    return [num_features, num_channels]