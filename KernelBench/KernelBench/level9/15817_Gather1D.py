import torch
import torch.onnx
import torch.nn as nn


class Gather1D(nn.Module):

    def forward(self, x):
        return x[[2, 4, 5]]


def get_inputs():
    return [torch.rand([6, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
