import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import *
from torch.nn.modules import *
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.distributed


class ContrastivePairwiseEmbeddingLoss(nn.Module):
    """
    ContrastivePairwiseEmbeddingLoss – proof of concept criterion.
    Still work in progress.
    """

    def __init__(self, margin=1.0, reduction='mean'):
        """
        Constructor method for the ContrastivePairwiseEmbeddingLoss class.
        Args:
            margin: margin parameter.
            reduction: criterion reduction type.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction or 'none'

    def forward(self, embeddings_pred, embeddings_true):
        """
        Work in progress.
        Args:
            embeddings_pred: predicted embeddings
            embeddings_true: true embeddings

        Returns:
            loss
        """
        device = embeddings_pred.device
        pairwise_similarity = torch.einsum('se,ae->sa', embeddings_pred,
            embeddings_true)
        bs = embeddings_pred.shape[0]
        batch_idx = torch.arange(bs, device=device)
        loss = F.cross_entropy(pairwise_similarity, batch_idx, reduction=
            self.reduction)
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
