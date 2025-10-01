import torch
import torch.nn as nn


class LearnableBias(nn.Module):

    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1),
            requires_grad=True)
        self.out_chn = out_chn

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_chn': 4}]
