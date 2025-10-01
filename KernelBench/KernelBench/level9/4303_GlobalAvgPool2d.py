import torch
import torch.nn as nn


class GlobalAvgPool2d(nn.Module):

    def forward(self, inputs):
        return inputs.mean(-1).mean(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
