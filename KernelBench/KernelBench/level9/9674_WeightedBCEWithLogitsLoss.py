import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted binary cross-entropy with logits.
    """

    def __init__(self, size_average=True, reduce=True, eps=0.0):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.eps = eps

    def forward(self, pred, target, weight_mask=None):
        return F.binary_cross_entropy_with_logits(pred, target.clamp(self.
            eps, 1 - self.eps), weight_mask)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
