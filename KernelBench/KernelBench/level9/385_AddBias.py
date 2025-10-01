import torch
import torch.nn as nn


class AddBias(nn.Module):

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self.bias = bias

    def forward(self, x):
        return x + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'bias': 4}]
