import torch
import numpy as np
from torch import nn


class pHAbsLayer(nn.Module):
    """Custom pHAbs Layer: Amax/(1+e^(pKa-pH)/phi)"""

    def __init__(self):
        super().__init__()
        weights = np.random.normal([1, 7.6, 0.5], [0.2, 0.5, 0.1])
        weights = torch.from_numpy(weights)
        self.weights = nn.Parameter(weights)
        self.regularizer = torch.zeros(3, dtype=torch.float64)

    def forward(self, x):
        y = self.weights[0] / (1 + torch.exp((self.weights[1] - x) / self.
            weights[2]))
        return y


class pHAbsModel(nn.Module):

    def __init__(self, lam_Amax=0, lam_pKa=0, lam_phi=0):
        super().__init__()
        self.f_pH = pHAbsLayer()
        self.f_pH.regularizer[0] = lam_Amax
        self.f_pH.regularizer[1] = lam_pKa
        self.f_pH.regularizer[2] = lam_phi

    def forward(self, x):
        return self.f_pH(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
