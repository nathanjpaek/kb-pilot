import torch
from abc import abstractmethod
import torch.utils.data.dataloader
import torch.nn.functional as F
import torch.nn as nn
import torch.nn


class SimilarityLoss(nn.Module):

    def __init__(self):
        super(SimilarityLoss, self).__init__()

    @abstractmethod
    def forward(self, inputs, targets):
        pass


class PairwiseBCELoss(SimilarityLoss):
    """
    Binary cross entropy between pair similarities and pair labels.
    """

    def __init__(self, balanced=False):
        super(PairwiseBCELoss, self).__init__()
        self.balanced = balanced

    def forward(self, inputs, targets):
        n = inputs.shape[0]
        neg_targets = torch.ones_like(targets) - targets
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets,
            reduction='none')
        if self.balanced:
            weight_matrix = n * (targets / 2.0 + neg_targets / (2.0 * (n - 1)))
            bce_loss *= weight_matrix
        loss = bce_loss.mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
