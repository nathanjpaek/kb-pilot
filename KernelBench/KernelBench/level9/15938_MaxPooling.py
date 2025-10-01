import torch
import torch.utils.data
import torch.nn as nn
import torch as torch


class MaxPooling(nn.Module):

    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, input):
        _b, _c, h, _w = input.size()
        f_pool = nn.MaxPool2d((h, 1), (1, 1))
        conv = f_pool(input)
        _b, _c, h, _w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        return conv


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
