import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs BMM, BatchNorm, Clamp, GroupNorm, LeakyReLU, Clamp.
    Operators used: BMM, BatchNorm, Clamp, GroupNorm, LeakyReLU, Clamp
    """
    def __init__(self, num_features, num_groups, bmm_dim):
        super(Model, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.group_norm = nn.GroupNorm(num_groups, num_features)
        self.leaky_relu = nn.LeakyReLU()
        self.bmm_dim = bmm_dim

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, bmm_dim, num_features).
            y (torch.Tensor): Input tensor of shape (batch_size, num_features, bmm_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_features, num_features).
        """
        x = torch.clamp(x, min=-1.0, max=1.0)
        bmm_result = torch.bmm(y, x)
        norm_result = self.batch_norm(bmm_result.transpose(1, 2)).transpose(1, 2)
        group_norm_result = self.group_norm(norm_result)
        leaky_relu_result = self.leaky_relu(group_norm_result)
        clamped_result = torch.clamp(leaky_relu_result, min=0.0, max=2.0)
        return clamped_result

def get_inputs():
    batch_size = 256
    num_features = 512
    bmm_dim = 256

    x = torch.randn(batch_size, bmm_dim, num_features)
    y = torch.randn(batch_size, num_features, bmm_dim)

    return [x, y]

def get_init_inputs():
    num_features = 512
    num_groups = 32
    bmm_dim = 256
    return [num_features, num_groups, bmm_dim]