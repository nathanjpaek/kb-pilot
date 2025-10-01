import torch
from torch import nn
from torch.nn import functional as F


class L1ExactPenaltyConstraintLoss(nn.Module):

    def __init__(self):
        super(L1ExactPenaltyConstraintLoss, self).__init__()

    def forward(self, x):
        gap_constraint = F.relu(x)
        return torch.norm(gap_constraint, p=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
