import torch
import torch.nn as nn


class scale_feature(nn.Module):

    def __init__(self, scale):
        super(scale_feature, self).__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
