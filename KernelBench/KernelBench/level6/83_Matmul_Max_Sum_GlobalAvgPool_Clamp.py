import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs Matmul, Max, Sum, GlobalAvgPool, and Clamp operations.
    Operators used: Matmul, Max, Sum, GlobalAvgPool, Clamp
    """
    def __init__(self, dim1, dim2, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.linear = nn.Linear(dim2, dim2)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim1, dim2).
            y (torch.Tensor): Input tensor of shape (batch_size, dim2, dim1).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = torch.matmul(x, y)
        x = torch.max(x, dim=2, keepdim=False)[0]
        x = torch.sum(x, dim=1, keepdim=True)
        x = F.adaptive_avg_pool2d(x.unsqueeze(2), (1, 1))
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x.squeeze()

def get_inputs():
    batch_size = 1024
    dim1 = 256
    dim2 = 256

    x = torch.randn(batch_size, dim1, dim2)
    y = torch.randn(batch_size, dim2, dim1)
    return [x, y]

def get_init_inputs():
    dim1 = 128
    dim2 = 256
    clamp_min = 0.0
    clamp_max = 1.0
    return [dim1, dim2, clamp_min, clamp_max]