import torch
import torch.onnx
import torch.nn as nn


class ReduceMax(nn.Module):

    def forward(self, x):
        return torch.max(x, -1, keepdim=True)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
