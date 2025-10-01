import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that combines Matmul, Sum, LogSumExp, HardSwish, ResidualAdd, and Hardtanh.
    Operators used: Matmul, Sum, HardSwish, LogSumExp, ResidualAdd, Hardtanh
    """
    def __init__(self, feature_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(feature_size, hidden_size)
        self.hardtanh = nn.Hardtanh(min_val=-0.5, max_val=0.5)

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_size).
            y (torch.Tensor): Input tensor of shape (batch_size, feature_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # Matmul
        matmul_output = torch.matmul(x, y.transpose(0, 1))

        # Sum
        sum_output = torch.sum(matmul_output, dim=1, keepdim=True)

        # HardSwish
        hardswish_output = F.hardswish(sum_output)

        # LogSumExp
        logsumexp_output = torch.logsumexp(hardswish_output, dim=1, keepdim=True)

        # ResidualAdd
        residual_add_output = logsumexp_output + 0.1 * torch.randn_like(logsumexp_output)

        # Hardtanh
        hardtanh_output = self.hardtanh(residual_add_output)

        return hardtanh_output

def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 256
    feature_size = 1024

    return [torch.randn(batch_size, feature_size), torch.randn(batch_size, feature_size)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    feature_size = 1024
    hidden_size = 512
    return [feature_size, hidden_size]