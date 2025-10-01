import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model combining BMM, BatchNorm, Clamp, GroupNorm, LeakyReLU.
    Operators used: BMM, BatchNorm, Clamp, GroupNorm, LeakyReLU, Clamp
    """
    def __init__(self, num_features, num_groups, bmm_dim):
        super(Model, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.group_norm = nn.GroupNorm(num_groups, num_features)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.bmm_dim = bmm_dim


    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, bmm_dim).
            y (torch.Tensor): Input tensor of shape (batch_size, bmm_dim, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_features, num_features).
        """
        x = self.batch_norm(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = self.leaky_relu(x)
        x = torch.bmm(x, y)
        x = self.group_norm(x)
        x = torch.clamp(x, min=-0.5, max=0.5)
        return x

def get_inputs():
    batch_size = 256
    num_features = 256
    bmm_dim = 128

    x = torch.randn(batch_size, num_features, bmm_dim)
    y = torch.randn(batch_size, bmm_dim, num_features)
    return [x, y]

def get_init_inputs():
    num_features = 256
    num_groups = 8
    bmm_dim = 128
    return [num_features, num_groups, bmm_dim]