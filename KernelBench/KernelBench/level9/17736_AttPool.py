import torch
from torch import nn
from torch.nn import functional as F


class AttPool(nn.Module):
    """
    Pool representations along a dimension with learned softmax scores.

    Args:
        input_size (int): Input size.
        dim (int): Dimension on which to apply the attention pooling.
    """

    def __init__(self, input_size, dim):
        super(AttPool, self).__init__()
        self.lin = nn.Linear(input_size, 1)
        self.dim = dim

    def forward(self, x):
        scores = F.softmax(self.lin(x), dim=self.dim)
        x = (scores * x).sum(dim=self.dim, keepdim=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'dim': 4}]
