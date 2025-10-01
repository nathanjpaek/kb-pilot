import torch
from itertools import chain as chain
import torch.utils.data
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class StendLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(StendLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        start_pred = output[:, 0]
        end_pred = output[:, 1]
        start_target = target[0]
        end_target = target[1]
        start_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        end_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        start_comp = start_loss(start_pred, start_target)
        end_comp = end_loss(end_pred, end_target)
        return start_comp + end_comp


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
