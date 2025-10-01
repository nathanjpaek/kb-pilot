import torch
import torch.nn as nn
from torch.autograd import *
import torch.nn.init


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def get_inputs():
    return [torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4],
        dtype=torch.int64), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
