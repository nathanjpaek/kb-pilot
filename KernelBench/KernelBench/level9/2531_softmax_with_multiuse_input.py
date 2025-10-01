import torch
import torch.nn as nn
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl


class softmax_with_multiuse_input(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = nn.Softmax(dim=-1)(x)
        x2 = x + x1
        return x1, x2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
