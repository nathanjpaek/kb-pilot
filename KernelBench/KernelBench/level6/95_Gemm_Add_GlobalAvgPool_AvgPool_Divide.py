import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Gemm, GlobalAvgPool, Add, GlobalAvgPool, AvgPool, Divide operations.
    Operators used: Gemm, GlobalAvgPool, Add, GlobalAvgPool, AvgPool, Divide
    """
    def __init__(self, in_features, out_features, kernel_size):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, 1, out_features//kernel_size**2).
        """
        x = self.gemm(x)
        x = x + y
        x = x.view(x.shape[0], int(x.shape[1]**(1/2)), int(x.shape[1]**(1/2)))
        x = torch.unsqueeze(x, dim=1)
        x = nn.AdaptiveAvgPool2d((x.shape[2], x.shape[3]))(x)
        x = self.avgpool(x)
        x = x / (torch.mean(x))
        return x


def get_inputs():
    """
    Returns a list of input tensors for the model.
    """
    batch_size = 2048
    in_features = 1024
    out_features = 1024

    return [torch.randn(batch_size, in_features), torch.randn(batch_size, out_features)]


def get_init_inputs():
    """
    Returns a list of initialization arguments for the model.
    """
    in_features = 1024
    out_features = 1024
    kernel_size = 2

    return [in_features, out_features, kernel_size]