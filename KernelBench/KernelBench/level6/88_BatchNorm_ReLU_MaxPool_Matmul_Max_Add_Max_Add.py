# mean_runtime: 0.794 ms
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Matmul, BatchNorm, ReLU, Add, Max, and MaxPool operations.
    Operators used: Matmul, BatchNorm, ReLU, Add, MaxPool
    """
    def __init__(self, num_features, num_channels, height, width, dim1, dim2):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dim1 = dim1
        self.dim2 = dim2
        self.relu = nn.ReLU()
        self.add_weight = nn.Parameter(torch.randn(1))

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).
            y (torch.Tensor): Input tensor of shape (batch_size, dim1, dim2)

        Returns:
            torch.Tensor: Output tensor after applying the operations.
        """
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.matmul(y, y.transpose(1,2)).max() + x.max() + self.add_weight
        return x


def get_inputs():
    batch_size = 256
    num_channels = 32
    height, width = 64, 64
    dim1 = 128
    dim2 = 64

    x = torch.randn(batch_size, num_channels, height, width)
    y = torch.randn(batch_size, dim1, dim2)
    return [x,y]

def get_init_inputs():
    num_features = 128
    num_channels = 32
    height, width = 64, 64
    dim1 = 128
    dim2 = 64
    return [num_features, num_channels, height, width, dim1, dim2]