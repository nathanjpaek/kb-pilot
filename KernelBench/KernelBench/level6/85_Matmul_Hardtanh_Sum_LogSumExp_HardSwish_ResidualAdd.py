import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model combining Matmul, Sum, LogSumExp, HardSwish, ResidualAdd, and Hardtanh.

    Operators Used:
    - Matmul
    - Hardtanh
    - Sum
    - LogSumExp
    - HardSwish
    - ResidualAdd
    """
    def __init__(self, feature_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(feature_size, hidden_size)
        self.hardtanh = nn.Hardtanh(-2, 2)
        self.feature_size = feature_size
        self.hidden_size = hidden_size

    def forward(self, x, residual):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_size).
            residual (torch.Tensor): Residual tensor of shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = torch.matmul(x, torch.randn(self.feature_size, self.hidden_size).to(x.device))
        x = self.hardtanh(x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = F.hardswish(x)
        x = x + residual
        return x


def get_inputs():
    batch_size = 2048
    feature_size = 256
    hidden_size = 128

    x = torch.randn(batch_size, feature_size)
    residual = torch.randn(batch_size, hidden_size)
    return [x, residual]

def get_init_inputs():
    feature_size = 256
    hidden_size = 128
    return [feature_size, hidden_size]