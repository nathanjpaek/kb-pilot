import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from math import sqrt as sqrt
from itertools import product as product


class MarginRankingLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        random = torch.randperm(inputs.size(0))
        inputs[random]
        pred_lossi = inputs[:inputs.size(0) // 2]
        pred_lossj = inputs[inputs.size(0) // 2:]
        target_loss = targets.reshape(inputs.size(0), 1)
        target_loss = target_loss[random]
        target_lossi = target_loss[:inputs.size(0) // 2]
        target_lossj = target_loss[inputs.size(0) // 2:]
        final_target = torch.sign(target_lossi - target_lossj)
        return F.margin_ranking_loss(pred_lossi, pred_lossj, final_target,
            margin=self.margin, reduction='mean')


def get_inputs():
    return [torch.rand([4, 1]), torch.rand([4, 1])]


def get_init_inputs():
    return [[], {}]
