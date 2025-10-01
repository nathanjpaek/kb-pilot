import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs Matmul, Sum, LogSumExp, HardSwish, ResidualAdd, and Hardtanh operations.

    Operators Used: Matmul, Sum, LogSumExp, HardSwish, ResidualAdd, Hardtanh
    """
    def __init__(self, feature_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(feature_size, hidden_size)
        self.hardtanh = nn.Hardtanh(-3, 3)

    def forward(self, x, residual):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_size).
            residual (torch.Tensor): Residual tensor of shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = self.linear(x)
        x = F.hardswish(x)
        x = torch.matmul(x, x.transpose(1, 0))
        x = torch.sum(x, dim=1)
        x = torch.logsumexp(x, dim=0)
        x = x + residual.sum()
        x = self.hardtanh(x)
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 256
    feature_size = 1024
    hidden_size = 512

    x = torch.randn(batch_size, feature_size)
    residual = torch.randn(batch_size, hidden_size)
    return [x, residual]


def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    feature_size = 1024
    hidden_size = 512
    return [feature_size, hidden_size]