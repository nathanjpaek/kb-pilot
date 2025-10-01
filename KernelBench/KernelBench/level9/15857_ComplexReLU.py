import torch
from torch import nn
import torch.utils


class ComplexReLU(nn.Module):

    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.relu_r = nn.ReLU()
        self.relu_i = nn.ReLU()

    def forward(self, input_r, input_i):
        return self.relu_r(input_r), self.relu_i(input_i)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
