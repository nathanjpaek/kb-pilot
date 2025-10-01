import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs Matmul, Max, Sum, GlobalAvgPool, and Clamp operations.
    Operators used: Matmul, Clamp, GlobalAvgPool, Sum, Max
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim).
            y (torch.Tensor): Input tensor of shape (batch_size, feature_dim, hidden_dim).

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = torch.matmul(x, y)
        x = torch.clamp(x, min=0)
        x = F.adaptive_avg_pool1d(x.transpose(1, 2), 1).squeeze(-1) 
        x = torch.sum(x, dim=1)
        x = torch.max(x, dim=0).values
        return x


def get_inputs():
    batch_size = 512
    seq_len = 128
    feature_dim = 256
    hidden_dim = 512

    x = torch.randn(batch_size, seq_len, feature_dim)
    y = torch.randn(batch_size, feature_dim, hidden_dim)
    return [x, y]

def get_init_inputs():
    return []