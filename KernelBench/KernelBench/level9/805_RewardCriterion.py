import torch
import torch.nn as nn
from torch.autograd import *


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, gpn_loss=None):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1),
            mask[:, :-1]], 1)).view(-1)
        if gpn_loss is None:
            output = -input * reward * mask
            output = torch.sum(output) / torch.sum(mask)
        else:
            gpn_loss = gpn_loss.unsqueeze(1).expand(gpn_loss.size(0), seq.
                size(1)).contiguous().view(-1)
            output = (-input * reward + gpn_loss * torch.exp(reward)) * mask
            output = torch.sum(output) / torch.sum(mask)
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
