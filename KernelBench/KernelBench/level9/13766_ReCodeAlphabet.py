import torch
import torch.nn as nn


class ReCodeAlphabet(nn.Module):

    def __init__(self):
        super(ReCodeAlphabet, self).__init__()

    def forward(self, input):
        input_reordered = [input[:, i, ...] for i in [0, 2, 1, 3]]
        input = torch.stack(input_reordered, dim=1)
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
