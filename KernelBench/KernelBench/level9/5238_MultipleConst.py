import torch
import torch.nn as nn


class MultipleConst(nn.Module):

    def forward(self, data):
        return 255 * data


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
