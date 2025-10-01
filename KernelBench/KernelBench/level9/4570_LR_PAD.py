import torch
import torch.nn as nn


def lr_pad(x, padding=1):
    """ Pad left/right-most to each other instead of zero padding """
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    """ Pad left/right-most to each other instead of zero padding """

    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
