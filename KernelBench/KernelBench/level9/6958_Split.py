import torch
import torch.nn as nn


class Split(nn.Module):

    def __init__(self):
        super(Split, self).__init__()

    def forward(self, x):
        n = int(x.size(1) / 2)
        x1 = x[:, :n, :, :].contiguous()
        x2 = x[:, n:, :, :].contiguous()
        return x1, x2

    def inverse(self, x1, x2):
        return torch.cat((x1, x2), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
