import torch
import torch.nn as nn
import torch.nn.functional as F


class CReLU(nn.ReLU):

    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, input):
        return torch.cat((F.relu(input, self.inplace), F.relu(-input, self.
            inplace)), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
