import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class BCE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logit, target, epoch=0):
        target = target.float()
        pred_prob = F.sigmoid(logit)
        return F.binary_cross_entropy(pred_prob, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
