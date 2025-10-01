import torch
import torch.nn as nn


class merge(nn.Module):

    def forward(self, x, y):
        return x + y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
