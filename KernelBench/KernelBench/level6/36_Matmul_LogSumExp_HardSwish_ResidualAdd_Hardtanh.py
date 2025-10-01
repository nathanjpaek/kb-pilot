import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    A model that combines Matmul, Sum, LogSumExp, HardSwish, ResidualAdd, and Hardtanh.
    Operators used: Matmul, LogSumExp, HardSwish, ResidualAdd, Hardtanh
    """
    def __init__(self, feature_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(feature_size, hidden_size)
        self.hardtanh = nn.Hardtanh(-1, 1)

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
        x = torch.matmul(x, y.transpose(0, 1))

        # LogSumExp
        x = torch.logsumexp(x, dim=1, keepdim=True)
        
        # HardSwish
        x = F.hardswish(x)

        # ResidualAdd
        residual = self.linear(y)
        x = x + residual

        # Hardtanh
        x = self.hardtanh(x)

        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 2048
    feature_size = 1024
    # in the format of a list of tensors randomized
    return [torch.randn(batch_size, feature_size), torch.randn(batch_size, feature_size)]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    feature_size = 1024
    hidden_size = 256
    # in the format of list[init args]
    return [feature_size, hidden_size]