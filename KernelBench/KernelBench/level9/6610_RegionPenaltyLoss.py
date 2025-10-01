import torch
from torch import nn


class RegionPenaltyLoss(nn.Module):

    def __init__(self, scale=1.0):
        """
        Multiplicative penalty.

        Penalizes "forbidden" regions instead of exact distribution matches.
        Optionally used in tandem with MTCrossEntropyRegionAwareLoss.
        `scale` param allows caller to scale the loss in order to match
        magnitude of other loss terms
        """
        super().__init__()
        self.scale = scale

    def forward(self, preds, targets):
        """
        """
        batch_size = preds.shape[0]
        penalty = torch.abs(targets - targets.max())
        penalty /= torch.sum(penalty)
        loss = preds * penalty
        loss = loss.view(batch_size, -1)
        return torch.sum(loss, dim=1) * self.scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
