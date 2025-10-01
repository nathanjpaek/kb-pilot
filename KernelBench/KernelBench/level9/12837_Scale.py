import math
import torch
import torch.nn as nn


class Scale(nn.Module):

    def __init__(self, d_model):
        super(Scale, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        return x * math.sqrt(self.d_model)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
