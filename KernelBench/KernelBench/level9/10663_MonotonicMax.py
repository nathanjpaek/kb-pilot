import torch
import torch.nn as nn


class MonotonicMax(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat(tuple(torch.max(i, dim=1)[0].unsqueeze(1) for i in
            x), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
