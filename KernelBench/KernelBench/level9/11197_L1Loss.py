import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F


class L1Loss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(L1Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
