import torch
import torch.onnx
import torch.nn as nn


class ReduceSum(nn.Module):

    def forward(self, x):
        return torch.sum(x, -1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
