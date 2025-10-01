import torch
from torch import nn
import torch.nn.functional as F


def cross_entropy(y, target, mask=None):
    if target.ndim == 1:
        loss = F.cross_entropy(y, target, reduction='none')
    else:
        loss = -(target * F.log_softmax(y, 1)).sum(1)
    if mask is not None:
        loss = mask * loss
    return loss.mean()


class CrossEntropy(nn.Module):

    def forward(self, y, target, mask=None, *args, **kwargs):
        return cross_entropy(y, target.detach(), mask)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
