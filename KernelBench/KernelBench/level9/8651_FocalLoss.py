import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    """Non weighted version of Focal Loss"""

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets,
            reduction='none')
        targets = targets.type(torch.long)
        targets.size(0)
        a = 2 * self.alpha - 1
        b = 1 - self.alpha
        at = targets * a + b
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
