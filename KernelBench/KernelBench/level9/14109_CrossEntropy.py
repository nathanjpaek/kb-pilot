import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel


def cross_entropy(y, target, mask=None):
    if len(target.shape) < 2:
        loss = F.cross_entropy(y, target, reduction='none')
    else:
        loss = -(target * F.log_softmax(y, 1)).sum(1)
    if mask is not None:
        loss = loss * mask
    return loss.mean()


class CrossEntropy(nn.Module):

    def forward(self, y, target, mask=None, *args, **kwargs):
        return cross_entropy(y, target.detach(), mask)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
