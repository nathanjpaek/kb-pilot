import torch
from torch import nn
from torch.nn import functional as F


class TVLoss(nn.Module):

    def forward(self, input):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
        y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
        diff = x_diff ** 2 + y_diff ** 2 + 1e-08
        return diff.mean(dim=1).sqrt().mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
