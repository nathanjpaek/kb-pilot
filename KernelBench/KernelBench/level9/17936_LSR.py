import torch
import torch.nn as nn
import torch.nn.functional as F


class LSR(nn.Module):

    def __init__(self, epsilon=0.1, num_classes=162):
        super(LSR, self).__init__()
        self._epsilon = epsilon
        self._num_classes = num_classes

    def forward(self, yhat, y):
        prior = torch.div(torch.ones_like(yhat), self._num_classes)
        loss = F.cross_entropy(yhat, y, reduction='none')
        reg = (-1 * F.log_softmax(yhat, dim=-1) * prior).sum(-1)
        total = (1 - self._epsilon) * loss + self._epsilon * reg
        lsr_loss = total.mean()
        return lsr_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
