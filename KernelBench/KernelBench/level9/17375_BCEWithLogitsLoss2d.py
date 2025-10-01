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


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
