import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout2d(nn.Dropout2d):

    def forward(self, input):
        return F.dropout2d(input, self.p, True, self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
