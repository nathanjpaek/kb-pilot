import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs scaling, matmul, softmax, hardswish, and max operations.
    Operators: Scaling, Matmul, Softmax, HardSwish, Max
    """
    def __init__(self, scale_factor, dim):
        super(Model, self).__init__()
        self.scale_factor = scale_factor
        self.dim = dim

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            y (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len).
        """
        x = x * self.scale_factor
        x = torch.matmul(x, y)
        x = F.softmax(x, dim=self.dim)
        x = F.hardswish(x)
        x = torch.max(x, dim=-1).values
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 1024
    seq_len = 256
    hidden_dim = 512

    x = torch.randn(batch_size, seq_len, hidden_dim)
    y = torch.randn(batch_size, hidden_dim, seq_len)
    return [x, y]

def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    scale_factor = 0.5
    dim = -1
    return [scale_factor, dim]