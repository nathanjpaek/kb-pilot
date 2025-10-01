import torch
import torch.nn as nn


class PoseNormalize(nn.Module):

    @torch.no_grad()
    def forward(self, x):
        return x * 2 - 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
