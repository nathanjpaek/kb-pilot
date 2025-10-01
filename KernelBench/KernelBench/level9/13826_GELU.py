import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch._utils
import torch.nn


class GELU(nn.Module):

    @staticmethod
    def forward(x):
        erf = F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        return 0.5 * x * (1 + erf)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
