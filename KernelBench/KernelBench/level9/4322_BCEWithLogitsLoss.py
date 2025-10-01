import torch
from torch import nn as nn
from torch.utils import data as data
from torch import autograd as autograd
import torch.onnx


class BCEWithLogitsLoss(nn.Module):

    def __init__(self, loss_weight=1.0, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.bce_wlogits_loss = nn.BCEWithLogitsLoss(**kwargs)
        self.loss_weight = loss_weight

    def forward(self, pred, gt):
        return self.bce_wlogits_loss(pred, gt) * self.loss_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
