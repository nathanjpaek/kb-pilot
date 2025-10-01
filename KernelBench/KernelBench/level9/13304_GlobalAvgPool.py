import torch
import torch.nn as nn


class GlobalAvgPool(nn.Module):

    def forward(self, x):
        N, C = x.size(0), x.size(1)
        return x.view(N, C, -1).mean(dim=2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
