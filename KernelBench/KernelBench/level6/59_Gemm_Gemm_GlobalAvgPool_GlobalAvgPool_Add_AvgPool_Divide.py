import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model utilizing Gemm, GlobalAvgPool, Add, AvgPool, and Divide operations.
    Operators used: Gemm, GlobalAvgPool, Add, AvgPool, Divide
    """
    def __init__(self, in_features, hidden_features, num_classes, pool_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        self.add_weight = nn.Parameter(torch.randn(hidden_features))
        self.avg_pool = nn.AvgPool2d(pool_size)
        self.num_classes = num_classes

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Gemm (Linear layer is equivalent to GEMM with bias)
        x = self.fc(x)
        y = self.fc(y)
        # GlobalAvgPool
        x = torch.mean(x, dim=0, keepdim=True)
        y = torch.mean(y, dim=0, keepdim=True)

        x = x.reshape(x.shape[0], int(x.shape[1]**0.5), int(x.shape[1]**0.5))
        y = y.reshape(y.shape[0], int(y.shape[1]**0.5), int(y.shape[1]**0.5))

        # Add
        added = x + y
        # AvgPool
        pooled = self.avg_pool(added)
        # Divide
        divided = pooled / (torch.sum(pooled) + 1e-5)
        return divided.reshape(1, -1)


def get_inputs():
    batch_size = 1024
    in_features = 1024

    return [torch.randn(batch_size, in_features), torch.randn(batch_size, in_features)]

def get_init_inputs():
    in_features = 1024
    hidden_features = 1024
    num_classes = 10
    pool_size = 8

    return [in_features, hidden_features, num_classes, pool_size]