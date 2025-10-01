import torch
import torch.nn as nn


class Softmax(nn.Module):

    def forward(self, x):
        y = torch.exp(x)
        return y / torch.sum(y, dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
