import torch
import torch.nn as nn
import torch.utils.data.distributed
from torch.backends import cudnn as cudnn


class BCEWithLogitsLoss2d(nn.Module):
    """Computationally stable version of 2D BCE loss.
    """

    def __init__(self):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(None, reduction='mean')

    def forward(self, logits, targets):
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(logits_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    """SoftJaccard loss for binary problems.
    """

    def forward(self, logits, labels):
        num = labels.size(0)
        m1 = torch.sigmoid(logits.view(num, -1))
        m2 = labels.view(num, -1)
        intersection = (m1 * m2).sum(1)
        score = (intersection + 1e-15) / (m1.sum(1) + m2.sum(1) + 1e-15)
        dice = score.sum(0) / num
        return 1 - dice


class BCEDiceLoss(torch.nn.Module):

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.dice = SoftDiceLoss()
        self.bce = BCEWithLogitsLoss2d()

    def forward(self, logits, targets):
        targets.size(0)
        loss = self.bce(logits, targets)
        loss += self.dice(logits, targets)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
