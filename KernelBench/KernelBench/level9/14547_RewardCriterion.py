import torch
import torch.nn as nn
from torch.autograd import *
import torch.nn


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, reduction='mean'):
        N, L = input.shape[:2]
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1),
            mask[:, :-1]], 1)).view(-1)
        output = -input * reward * mask
        if reduction == 'none':
            output = output.view(N, L).sum(1) / mask.view(N, L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)
        return output


def get_inputs():
    return [torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4],
        dtype=torch.int64), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
