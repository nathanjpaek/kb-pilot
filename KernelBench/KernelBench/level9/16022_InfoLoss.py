import math
import torch
import torch.nn as nn


class InfoLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-08):
        x = torch.mean(x, 0)
        logN = math.log(float(x.shape[0]))
        x = x * (x + eps).log() / logN
        neg_entropy = x.sum()
        return 1.0 + neg_entropy


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
