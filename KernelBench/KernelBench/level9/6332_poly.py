import torch
import numpy as np
import torch.nn as nn
import torch.utils.data


class poly(nn.Module):
    """Polynomial activation function. 
    degreelist: list of powers of the polynomial.
    """

    def __init__(self, degreelist):
        super(poly, self).__init__()
        self.degreelist = degreelist
        p = len(degreelist)
        arr = np.ones(p, dtype=np.float32)
        coeff = torch.nn.Parameter(torch.tensor(arr), requires_grad=True)
        self.register_parameter('coefficients', coeff)

    def forward(self, x):
        out = [torch.pow(x, n) for n in self.degreelist]
        shape = x.shape
        out = torch.cat([j.reshape(*shape, 1) for j in out], dim=-1)
        out = out * self.coefficients
        out = out.sum(-1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'degreelist': [4, 4]}]
