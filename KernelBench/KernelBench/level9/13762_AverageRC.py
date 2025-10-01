import torch
import torch.nn as nn


class AverageRC(nn.Module):

    def __init__(self):
        super(AverageRC, self).__init__()

    def forward(self, input):
        input = input[:int(input.shape[0] / 2)] / 2 + input[int(input.shape
            [0] / 2):] / 2
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
