import math
import torch
from torch import nn
import torch.utils.data


class JSD(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-08):
        logN = math.log(float(x.shape[0]))
        y = torch.mean(x, 0)
        y = y * (y + eps).log() / logN
        y = y.sum()
        x = x * (x + eps).log() / logN
        x = x.sum(1).mean()
        return 1.0 - x + y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
