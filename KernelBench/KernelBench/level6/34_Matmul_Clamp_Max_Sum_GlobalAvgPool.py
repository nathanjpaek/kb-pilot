import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Matmul, Max, Sum, GlobalAvgPool, and Clamp operations.
    Operators: Matmul, Clamp, Max, Sum, GlobalAvgPool
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
            torch.Tensor: Output tensor of shape (batch_size, 1, 1).
        """
        x = torch.matmul(x, y)
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.mean(x, dim=[1, 2])
        return x.unsqueeze(1).unsqueeze(1)


def get_inputs():
    """
    Returns example input tensors for the model.

    Returns:
        list[torch.Tensor]: List containing input tensors x and y.
    """
    batch_size = 256
    dim1 = 256
    dim2 = 512
    return [torch.randn(batch_size, dim1, dim2), torch.randn(batch_size, dim2, dim1)]

def get_init_inputs():
    """
    Returns initialization parameters for the model.

    Returns:
        list[int, int, float, float]: List containing dim1, dim2, clamp_min, and clamp_max.
    """
    dim1 = 256
    dim2 = 512
    clamp_min = -0.5
    clamp_max = 0.5
    return [dim1, dim2, clamp_min, clamp_max]