import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model using Matmul, BatchNorm, ReLU, Add, Max, and MaxPool.
    Operators Used: Matmul, BatchNorm, ReLU, MaxPool, Add, Max
    """
    def __init__(self, num_features, num_channels):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, seq_len).
            y (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size).
        """
        x = torch.matmul(x, y)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        x = x + 1
        x = torch.max(x, dim=2)[0]
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 256
    num_features = 512
    seq_len = 256

    return [torch.randn(batch_size, num_features, seq_len), torch.randn(batch_size, seq_len, num_features)]


def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    num_features = 512
    num_channels = 64
    return [num_features, num_channels]