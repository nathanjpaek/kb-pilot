import torch
import numpy as np
import torch.nn as nn


def cosine_basis_functions(x, n_basis_functions=64):
    """Cosine basis functions used to embed quantile thresholds.

    Args:
        x (torch.Tensor): Input.
        n_basis_functions (int): Number of cosine basis functions.

    Returns:
        ndarray: Embedding with shape of (x.shape + (n_basis_functions,)).
    """
    i_pi = torch.arange(1, n_basis_functions + 1, dtype=torch.float, device
        =x.device) * np.pi
    embedding = torch.cos(x[..., None] * i_pi)
    assert embedding.shape == x.shape + (n_basis_functions,)
    return embedding


class CosineBasisLinear(nn.Module):
    """Linear layer following cosine basis functions.

    Args:
        n_basis_functions (int): Number of cosine basis functions.
        out_size (int): Output size.
    """

    def __init__(self, n_basis_functions, out_size):
        super().__init__()
        self.linear = nn.Linear(n_basis_functions, out_size)
        self.n_basis_functions = n_basis_functions
        self.out_size = out_size

    def forward(self, x):
        """Evaluate.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output with shape of (x.shape + (out_size,)).
        """
        h = cosine_basis_functions(x, self.n_basis_functions)
        h = h.reshape(-1, self.n_basis_functions)
        out = self.linear(h)
        out = out.reshape(*x.shape, self.out_size)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_basis_functions': 4, 'out_size': 4}]
