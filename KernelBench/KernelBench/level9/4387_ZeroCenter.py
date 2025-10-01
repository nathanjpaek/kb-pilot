import torch
import torch.nn as nn


class ZeroCenter(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """x : [B, C, H, W]"""
        return x.sub_(x.flatten(1).mean(1, keepdim=True).unsqueeze(-1).
            unsqueeze(-1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
