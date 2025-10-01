import torch
import torch.nn as nn


class AdaptiveMaxPool2d(nn.Module):

    def __init__(self):
        super(AdaptiveMaxPool2d, self).__init__()
        self.layer = nn.AdaptiveMaxPool2d((5, 7))

    def forward(self, x):
        x = self.layer(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
