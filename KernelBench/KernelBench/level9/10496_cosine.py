import torch
import torch as tr
import torch.nn as nn


class cosine(nn.Module):

    def __init__(self):
        super(cosine, self).__init__()

    def forward(self, x: 'tr.Tensor'):
        return tr.cos(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
