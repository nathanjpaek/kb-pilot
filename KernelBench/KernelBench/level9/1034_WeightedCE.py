import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class WeightedCE(nn.Module):
    """Mask weighted multi-class cross-entropy (CE) loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=None):
        loss = F.cross_entropy(pred, target, reduction='none')
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
