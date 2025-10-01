import torch
import torch.utils.data
import torch.nn as nn


class RerangeLayer(nn.Module):

    def __init__(self):
        super(RerangeLayer, self).__init__()

    def forward(self, inp):
        return (inp + 1.0) / 2.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
