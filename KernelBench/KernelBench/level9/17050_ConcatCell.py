import torch
import torch as th
import torch.nn as nn


class ConcatCell(nn.Module):

    def __init__(self, input_dim):
        super(ConcatCell, self).__init__()
        self.input_dim = input_dim

    def forward(self, x1, x2):
        return th.cat([x1, x2], dim=-1)

    def get_output_dim(self):
        return self.input_dim * 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
