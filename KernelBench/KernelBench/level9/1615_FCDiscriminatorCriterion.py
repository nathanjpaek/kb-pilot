import torch
import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminatorCriterion(nn.Module):

    def __init__(self):
        super(FCDiscriminatorCriterion, self).__init__()

    def forward(self, pred, gt):
        loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        return torch.mean(loss, dim=(1, 2, 3))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
