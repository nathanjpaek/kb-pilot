import math
import torch
import torch.nn as nn


class EntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-08):
        logN = math.log(float(x.shape[0]))
        x = x * (x + eps).log() / logN
        neg_entropy = x.sum(1)
        return -neg_entropy.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
