import torch
import torch.nn as nn


class MultiRelu(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.relu1 = nn.ReLU(inplace=inplace)
        self.relu2 = nn.ReLU(inplace=inplace)

    def forward(self, arg1, arg2):
        return self.relu1(arg1), self.relu2(arg2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
