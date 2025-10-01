import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum(
        ) if reduction == 'sum' else loss


class FocalLoss(nn.Module):
    """
    Origianl code is from https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L226-L266
    """

    def __init__(self, alpha, gamma, normalize):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, preds, targets):
        cross_entropy = F.binary_cross_entropy_with_logits(preds, targets,
            reduction='none')
        gamma = self.gamma
        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = th.exp(-gamma * targets * preds - gamma * th.log1p(
                th.exp(-1.0 * preds)))
        loss = modulator * cross_entropy
        weighted_loss = self.alpha * loss
        focal_loss = reduce_loss(weighted_loss, reduction='sum')
        return focal_loss / targets.sum() if self.normalize else focal_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4, 'gamma': 4, 'normalize': 4}]
