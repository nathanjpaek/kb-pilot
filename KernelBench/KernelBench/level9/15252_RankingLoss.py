import torch
import torch.nn.functional as F
from abc import abstractmethod
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn


class SimilarityLoss(nn.Module):

    def __init__(self):
        super(SimilarityLoss, self).__init__()

    @abstractmethod
    def forward(self, inputs, targets):
        pass


class RankingLoss(SimilarityLoss):
    """
    Triplet ranking loss between pair similarities and pair labels.
    """

    def __init__(self, margin=0.1, direction_weights=[0.5, 0.5]):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.direction_weights = direction_weights

    def forward(self, inputs, targets):
        n = inputs.shape[0]
        neg_targets = torch.ones_like(targets) - targets
        ranking_loss_matrix_01 = neg_targets * F.relu(self.margin + inputs -
            torch.diag(inputs).view(n, 1))
        ranking_loss_matrix_10 = neg_targets * F.relu(self.margin + inputs -
            torch.diag(inputs).view(1, n))
        neg_targets_01_sum = torch.sum(neg_targets, dim=1)
        neg_targets_10_sum = torch.sum(neg_targets, dim=0)
        loss = self.direction_weights[0] * torch.mean(torch.sum(
            ranking_loss_matrix_01 / neg_targets_01_sum, dim=1)
            ) + self.direction_weights[1] * torch.mean(torch.sum(
            ranking_loss_matrix_10 / neg_targets_10_sum, dim=0))
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
