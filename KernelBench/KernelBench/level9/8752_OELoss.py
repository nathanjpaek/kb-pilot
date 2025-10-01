import torch
import torch.nn as nn


class OELoss(nn.Module):

    def __init__(self):
        super(OELoss, self).__init__()

    def forward(self, x):
        return -(x.mean(1) - torch.logsumexp(x, dim=1)).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
