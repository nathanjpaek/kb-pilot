import torch
import numpy as np
import torch.nn as nn


class DotProd(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, s, t):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        out = (s * t[:, None, :]).sum(dim=2)[:, 0]
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
