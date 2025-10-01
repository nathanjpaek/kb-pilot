import torch
import torch.nn as nn
from torch.autograd import *


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = seq > 0
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1
            ).reshape(-1)
        output = -input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def get_inputs():
    return [torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4],
        dtype=torch.int64), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
