import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel6_MultiTensor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        input = input1 + input2
        return 1 - F.relu(1 - input)[:, 1]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
