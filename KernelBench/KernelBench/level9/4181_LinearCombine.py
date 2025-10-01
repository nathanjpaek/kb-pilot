import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F


class LinearCombine(nn.Module):

    def __init__(self, layers_num, trainable=True, input_aware=False,
        word_level=False):
        super(LinearCombine, self).__init__()
        self.input_aware = input_aware
        self.word_level = word_level
        if input_aware:
            raise NotImplementedError('Input aware is not supported.')
        self.w = nn.Parameter(torch.full((layers_num, 1, 1, 1), 1.0 /
            layers_num), requires_grad=trainable)

    def forward(self, seq):
        nw = F.softmax(self.w, dim=0)
        seq = torch.mul(seq, nw)
        seq = torch.sum(seq, dim=0)
        return seq


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'layers_num': 1}]
