import torch
import torch.nn as nn


class MultConst(nn.Module):

    def forward(self, input):
        return 255 * input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
