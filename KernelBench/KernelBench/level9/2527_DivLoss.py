import torch
from torch import nn
from torch.optim.lr_scheduler import *


class DivLoss(nn.Module):

    def __init__(self):
        super(DivLoss, self).__init__()

    def forward(self, scores):
        mu = scores.mean(0)
        std = ((scores - mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt(
            )
        loss_std = -std.sum()
        return loss_std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
