import torch
import torch.nn as nn


class GramMatrix(nn.Module):

    def forward(self, input):
        _, channels, h, w = input.size()
        out = input.view(-1, h * w)
        out = torch.mm(out, out.t())
        return out.div(channels * h * w)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
