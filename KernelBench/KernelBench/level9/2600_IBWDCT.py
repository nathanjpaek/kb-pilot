import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from torch import nn
import torch.fft


class IBWDCT(nn.Module):

    def __init__(self):
        super().__init__()
        self.ibwdct = nn.ConvTranspose2d(64, 1, 8, 8, bias=False)
        self.ibwdct.weight.requires_grad = False
        for m in range(8):
            for n in range(8):
                for p in range(8):
                    for q in range(8):
                        self.ibwdct.weight[p * 8 + q, 0, m, n] = np.cos(np.
                            pi * (2 * m + 1) * p / 16) * np.cos(np.pi * (2 *
                            n + 1) * q / 16) * (np.sqrt(1 / 8) if p == 0 else
                            1 / 2) * (np.sqrt(1 / 8) if q == 0 else 1 / 2)

    def forward(self, x):
        return self.ibwdct(x)


def get_inputs():
    return [torch.rand([4, 64, 4, 4])]


def get_init_inputs():
    return [[], {}]
