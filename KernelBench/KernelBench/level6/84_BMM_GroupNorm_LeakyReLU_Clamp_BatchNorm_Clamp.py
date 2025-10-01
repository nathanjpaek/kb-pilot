import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs BMM, GroupNorm, LeakyReLU, Clamp, BatchNorm, Clamp operations.
    Operators used: BMM, GroupNorm, LeakyReLU, Clamp, BatchNorm, Clamp
    """
    def __init__(self, num_groups, num_channels, batch_size, seq_len, hidden_dim):
        super(Model, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.clamp1 = lambda x: torch.clamp(x, min=-1.0, max=1.0)
        self.batch_norm = nn.BatchNorm1d(seq_len * hidden_dim)
        self.clamp2 = lambda x: torch.clamp(x, min=-0.5, max=0.5)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            y (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        x = torch.bmm(x, y)
        x = self.group_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.leaky_relu(x)
        x = self.clamp1(x)
        x = x.reshape(self.batch_size, -1)
        x = self.batch_norm(x)
        x = x.reshape(self.batch_size, self.seq_len, self.hidden_dim)
        x = self.clamp2(x)
        return x

def get_inputs():
    batch_size = 2048
    seq_len = 32
    hidden_dim = 64

    x = torch.randn(batch_size, seq_len, hidden_dim)
    y = torch.randn(batch_size, hidden_dim, hidden_dim)
    return [x, y]

def get_init_inputs():
    num_groups = 8
    num_channels = 64
    batch_size = 2048
    seq_len = 32
    hidden_dim = 64
    return [num_groups, num_channels, batch_size, seq_len, hidden_dim]