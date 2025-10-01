import torch
import torch.nn as nn
import torch.utils


class CatCombine(nn.Module):

    def __init__(self, C):
        super(CatCombine, self).__init__()
        self.compress = nn.Linear(C * 2, C)

    def forward(self, x, y):
        return self.compress(torch.cat((x, y), dim=-1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C': 4}]
