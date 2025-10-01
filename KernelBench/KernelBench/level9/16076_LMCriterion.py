import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


class LMCriterion(nn.Module):

    def __init__(self):
        super(LMCriterion, self).__init__()

    def forward(self, input, target):
        logprob_select = torch.gather(input, 1, target)
        mask = target.data.gt(0)
        if isinstance(input, Variable):
            mask = Variable(mask, volatile=input.volatile)
        out = torch.masked_select(logprob_select, mask)
        loss = -torch.sum(out)
        return loss


def get_inputs():
    return [torch.ones([4, 4], dtype=torch.int64), torch.ones([4, 4], dtype
        =torch.int64)]


def get_init_inputs():
    return [[], {}]
