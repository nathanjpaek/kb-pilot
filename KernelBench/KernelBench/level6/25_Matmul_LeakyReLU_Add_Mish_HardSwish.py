import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs a Matmul, Swish, LeakyReLU, Mish, HardSwish, and Add operations.
    Operators: ['Matmul', 'Swish', 'LeakyReLU', 'Mish', 'HardSwish', 'Add']
    """
    def __init__(self, dim1, dim2, dim3):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(dim2, dim3))

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim1, dim2).
            y (torch.Tensor): Input tensor of shape (batch_size, dim1, dim3).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim1, dim3).
        """
        x = torch.matmul(x, self.weight)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = x + y
        x = x * torch.tanh(F.softplus(x)) # Mish activation
        x = x * F.hardtanh(x + 3, 0., 6.) / 6. + 0.5 # HardSwish
        return x


def get_inputs():
    batch_size = 256
    dim1 = 128
    dim2 = 256
    dim3 = 512

    x = torch.randn(batch_size, dim1, dim2)
    y = torch.randn(batch_size, dim1, dim3)
    return [x, y]

def get_init_inputs():
    dim1 = 128
    dim2 = 256
    dim3 = 512
    return [dim1, dim2, dim3]