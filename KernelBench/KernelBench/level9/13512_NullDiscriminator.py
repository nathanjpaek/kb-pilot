import torch
import torch.nn as nn
import torch.utils.data


class NullDiscriminator(nn.Module):

    def __init__(self):
        super(NullDiscriminator, self).__init__()

    def forward(self, inputs, y=None):
        d = inputs.sum(1, keepdim=True)
        return d


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
