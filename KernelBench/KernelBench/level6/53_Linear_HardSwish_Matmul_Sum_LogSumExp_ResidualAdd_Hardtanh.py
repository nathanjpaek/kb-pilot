import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model using Matmul, Sum, LogSumExp, HardSwish, ResidualAdd, Hardtanh.
    Operators used: Matmul, HardSwish, Sum, LogSumExp, ResidualAdd, Hardtanh
    """
    def __init__(self, in_features, hidden_dim, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, hidden_dim)
        self.weight = nn.Parameter(torch.randn(hidden_dim, out_features))
        self.hardtanh = nn.Hardtanh(min_val=-0.5, max_val=0.5)

    def forward(self, x, residual):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            residual (torch.Tensor): Residual tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.linear(x)
        x = F.hardswish(x)
        x = torch.matmul(x, self.weight)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x + residual
        x = self.hardtanh(x)
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 2048
    in_features = 1024
    out_features = 512

    return [torch.randn(batch_size, in_features), torch.randn(batch_size, out_features)]


def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_features = 1024
    hidden_dim = 512
    out_features = 512
    return [in_features, hidden_dim, out_features]