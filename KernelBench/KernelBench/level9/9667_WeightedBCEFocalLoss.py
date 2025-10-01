import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


class WeightedBCEFocalLoss(nn.Module):
    """Weighted binary focal loss with logits.
    """

    def __init__(self, gamma=2.0, alpha=0.25, eps=0.0):
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, weight_mask=None):
        pred_sig = pred.sigmoid()
        pt = (1 - target) * (1 - pred_sig) + target * pred_sig
        at = (1 - self.alpha) * target + self.alpha * (1 - target)
        wt = at * (1 - pt) ** self.gamma
        if weight_mask is not None:
            wt *= weight_mask
        bce = F.binary_cross_entropy_with_logits(pred, target.clamp(self.
            eps, 1 - self.eps), reduction='none')
        return (wt * bce).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
