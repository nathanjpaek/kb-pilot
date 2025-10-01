import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torch.optim


class Tanh2(nn.Module):

    def __init__(self):
        super(Tanh2, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return (self.tanh(x) + 1) / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
