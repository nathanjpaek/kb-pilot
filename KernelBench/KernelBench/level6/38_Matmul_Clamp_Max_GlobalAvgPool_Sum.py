import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs a Matmul, clamps the values, calculates the Max, applies GlobalAvgPool, and calculates the Sum.
    Operators used: Matmul, Clamp, Max, GlobalAvgPool, Sum
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            y (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, num_features).

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.matmul(x, y)
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = torch.max(x, dim=-1, keepdim=True)[0]
        x = F.adaptive_avg_pool1d(x.transpose(1,2), 1).squeeze(-1)
        x = torch.sum(x, dim=-1)
        return x


def get_inputs():
    batch_size = 256
    seq_len = 128
    hidden_dim = 256
    num_features = 512

    x = torch.randn(batch_size, seq_len, hidden_dim)
    y = torch.randn(batch_size, hidden_dim, num_features)
    return [x, y]

def get_init_inputs():
    return []