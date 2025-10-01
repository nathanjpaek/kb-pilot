import torch
import torch.nn as nn
import torch.utils.data


class DivLoss(nn.Module):

    def __init__(self):
        super(DivLoss, self).__init__()

    def forward(self, lam):
        mu = lam.mean(0)
        std = ((lam - mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
        loss_std = -std.sum()
        return loss_std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
