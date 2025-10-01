import torch
from torch import nn
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):

    def __init__(self, alpha=-1, gamma=2.0, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets,
            reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * (1 - p_t) ** self.gamma
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
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
