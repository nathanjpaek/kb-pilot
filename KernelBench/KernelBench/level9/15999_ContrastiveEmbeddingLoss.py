import torch
from torch.nn.modules.loss import *
import torch.nn as nn
from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *


class ContrastiveEmbeddingLoss(nn.Module):
    """
    Contrastive embedding loss

    paper: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0, reduction='mean'):
        """
        Constructor method for the ContrastiveEmbeddingLoss class.
        Args:
            margin: margin parameter.
            reduction: criterion reduction type.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or 'none'

    def forward(self, embeddings_left, embeddings_right, distance_true):
        """
        Forward propagation method for the contrastive loss.
        Args:
            embeddings_left: left objects embeddings
            embeddings_right: right objects embeddings
            distance_true: true distances

        Returns:
            loss
        """
        diff = embeddings_left - embeddings_right
        distance_pred = torch.sqrt(torch.sum(torch.pow(diff, 2), 1))
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
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
