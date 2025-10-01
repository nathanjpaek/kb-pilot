import torch
import torch.nn as nn
from torch.nn.modules.loss import *
from torch.nn.modules import *
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.distributed


class ContrastiveDistanceLoss(nn.Module):
    """
    Contrastive distance loss
    """

    def __init__(self, margin=1.0, reduction='mean'):
        """
        Constructor method for the ContrastiveDistanceLoss class.
        Args:
            margin: margin parameter.
            reduction: criterion reduction type.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or 'none'

    def forward(self, distance_pred, distance_true):
        """
        Forward propagation method for the contrastive loss.
        Args:
            distance_pred: predicted distances
            distance_true: true distances

        Returns:
            loss
        """
        bs = len(distance_true)
        margin_distance = self.margin - distance_pred
        margin_distance_ = torch.clamp(margin_distance, min=0.0)
        loss = (1 - distance_true) * torch.pow(distance_pred, 2
            ) + distance_true * torch.pow(margin_distance_, 2)
        if self.reduction == 'mean':
            loss = torch.sum(loss) / 2.0 / bs
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
