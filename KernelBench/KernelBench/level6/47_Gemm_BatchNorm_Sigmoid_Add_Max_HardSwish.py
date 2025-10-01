import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a Gemm, BatchNorm, Sigmoid, Add, Max, and HardSwish operations.
    Operators used: Gemm, BatchNorm, Sigmoid, Add, Max, HardSwish
    """
    def __init__(self, in_features, out_features, batch_size):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Second input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = torch.matmul(x, self.weight.T) + self.bias # Gemm
        x = self.bn(x) # BatchNorm
        x = torch.sigmoid(x) # Sigmoid
        x = x + y # Add
        x = torch.max(x, dim=1, keepdim=True)[0] # Max
        x = x * (torch.clamp(x + 3, 0, 6) / 6) # HardSwish
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 2048
    in_features = 2048
    out_features = 1024

    x = torch.randn(batch_size, in_features)
    y = torch.randn(batch_size, out_features)
    return [x, y]


def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_features = 2048
    out_features = 1024
    batch_size = 2048
    return [in_features, out_features, batch_size]