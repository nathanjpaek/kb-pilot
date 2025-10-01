import torch
import torch.onnx
import torch.nn as nn


class Asinh(nn.Module):

    def forward(self, x):
        return torch.asinh(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
