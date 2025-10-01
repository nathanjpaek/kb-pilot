import torch
import torch.utils.data
import torch.nn as nn


class My_Tanh(nn.Module):

    def __init__(self):
        super(My_Tanh, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 0.5 * (self.tanh(x) + 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
