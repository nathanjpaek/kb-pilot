import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss_learning_loss(nn.Module):
    """
    Ranking loss as described in LPM paper
    inputs/targets are randomly permutated 
    final target is a list of -1 and 1's 
    -1 means the item in the i list is higher 1 means the item in the j list is higher
    This creates a pairwise ranking loss
    """

    def __init__(self, margin=0.5):
        super(MarginRankingLoss_learning_loss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        random = torch.randperm(inputs.size(0))
        mid = int(inputs.size(0) // 2)
        pred_lossi = inputs[:mid]
        pred_lossj = inputs[mid:]
        target_loss = targets.reshape(inputs.size(0), 1)
        target_loss = target_loss[random]
        target_lossi = target_loss[:mid]
        target_lossj = target_loss[mid:]
        final_target = torch.sign(target_lossi - target_lossj)
        return F.margin_ranking_loss(pred_lossi, pred_lossj, final_target,
            margin=self.margin, reduction='mean')


def get_inputs():
    return [torch.rand([4, 1]), torch.rand([4, 1])]


def get_init_inputs():
    return [[], {}]
