import torch
import torch.nn as nn


class interaction(nn.Module):

    def __init__(self, conf):
        super().__init__()

    def forward(self, p, h):
        p = p.unsqueeze(2)
        h = h.unsqueeze(1)
        return p * h


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'conf': 4}]
