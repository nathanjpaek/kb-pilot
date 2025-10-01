import torch
import torch.nn as nn
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl


class inplace_softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = x + 1
        x2 = nn.Softmax(dim=-1)(x1)
        return x2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
