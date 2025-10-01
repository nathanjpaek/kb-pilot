import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        out = -1.0 * out.sum(dim=1)
        return out.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
