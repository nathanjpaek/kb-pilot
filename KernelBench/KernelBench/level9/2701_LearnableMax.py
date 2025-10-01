import torch
import torch.nn as nn


class LearnableMax(nn.Module):

    def __init__(self, out_chn):
        super(LearnableMax, self).__init__()
        self.max1 = nn.Parameter(torch.zeros(1, out_chn, 1, 1),
            requires_grad=True)
        self.max2 = nn.Parameter(torch.zeros(1, out_chn, 1, 1),
            requires_grad=True)
        self.out_chn = out_chn * 2

    def forward(self, x):
        out = torch.max(x, self.max1.expand_as(x))
        out = torch.max(out, self.max2.expand_as(x))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_chn': 4}]
