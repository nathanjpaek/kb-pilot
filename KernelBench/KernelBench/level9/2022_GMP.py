import torch


class GMP(torch.nn.Module):
    """A global max pooling module.

    Args:
        dim (int): The dimension at which to compute the maximum.
    """

    def __init__(self, dim: 'int'):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
