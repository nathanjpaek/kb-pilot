import torch
import torch.nn.functional as F
import torch.nn as nn


class SigmoidFocalLoss(nn.Module):

    def __init__(self, alpha: 'float'=-1, gamma: 'float'=2, reduction:
        'str'='none'):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target,
            reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * (1 - p_t) ** self.gamma
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
