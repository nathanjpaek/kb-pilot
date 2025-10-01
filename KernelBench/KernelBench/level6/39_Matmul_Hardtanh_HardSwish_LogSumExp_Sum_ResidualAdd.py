import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model using Matmul, Sum, LogSumExp, HardSwish, ResidualAdd, and Hardtanh.

    Operators used:
    - Matmul
    - Hardtanh
    - HardSwish
    - LogSumExp
    - Sum
    - ResidualAdd
    """
    def __init__(self, feature_dim, hidden_dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(feature_dim, hidden_dim)
        self.hardtanh = nn.Hardtanh(min_val=-0.5, max_val=0.5)

    def forward(self, x, residual):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_dim).
            residual (torch.Tensor): Residual tensor of shape (batch_size, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_dim).
        """
        x = torch.matmul(x, self.linear.weight.T)
        x = self.hardtanh(x)
        x = F.hardswish(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.sum(x, dim=1, keepdim=True)
        x = x + residual
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 2048
    feature_dim = 2048
    hidden_dim = 1024
    return [torch.randn(batch_size, feature_dim), torch.randn(batch_size, hidden_dim)]


def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    feature_dim = 2048
    hidden_dim = 1024
    return [feature_dim, hidden_dim]