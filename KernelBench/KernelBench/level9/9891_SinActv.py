import torch
import torch.nn as nn


class SinActv(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_):
        return torch.sin(input_)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
