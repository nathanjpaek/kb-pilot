import torch
import torch.nn as nn


class Sine(nn.Module):

    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(5 * input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
