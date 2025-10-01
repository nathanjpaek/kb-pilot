import torch
import numpy as np
import torch.nn as nn


class TimeEncode(nn.Module):
    """Use finite fourier series with different phase and frequency to encode
    time different between two event

    ..math::
        \\Phi(t) = [\\cos(\\omega_0t+\\psi_0),\\cos(\\omega_1t+\\psi_1),...,\\cos(\\omega_nt+\\psi_n)] 

    Parameter
    ----------
    dimension : int
        Length of the fourier series. The longer it is , 
        the more timescale information it can capture

    Example
    ----------
    >>> tecd = TimeEncode(10)
    >>> t = torch.tensor([[1]])
    >>> tecd(t)
    tensor([[[0.5403, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000]]], dtype=torch.float64, grad_fn=<CosBackward>)
    """

    def __init__(self, dimension):
        super(TimeEncode, self).__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.w.weight = torch.nn.Parameter(torch.from_numpy(1 / 10 ** np.
            linspace(0, 9, dimension)).float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        t = t.unsqueeze(dim=2).float()
        output = torch.cos(self.w(t))
        return output


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dimension': 4}]
