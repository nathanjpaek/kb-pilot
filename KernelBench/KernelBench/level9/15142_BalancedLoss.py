import torch
from torch import nn
import torch.nn.functional as F


class BalancedLoss(nn.Module):

    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight

    def forward(self, input, target):
        pos_mask = target == 0
        neg_mask = target == 1
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = torch.zeros(target.size(), dtype=torch.float32, device=
            target.device)
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        return F.binary_cross_entropy_with_logits(input, target.float(),
            weight, reduction='sum')


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
