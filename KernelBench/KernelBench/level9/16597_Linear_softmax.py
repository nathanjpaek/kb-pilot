import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_softmax(nn.Module):

    def __init__(self, inp, out):
        super(Linear_softmax, self).__init__()
        self.f1 = nn.Linear(inp, out)

    def forward(self, x):
        x = self.f1(x)
        return F.softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inp': 4, 'out': 4}]
