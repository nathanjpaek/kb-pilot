import torch
import torch.nn as nn


class L2Norm(nn.Module):

    def forward(self, x, eps=1e-06):
        norm = x.norm(dim=1, keepdim=True).clamp(min=eps)
        return x / norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
