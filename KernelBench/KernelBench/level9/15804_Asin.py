import torch
import torch.onnx
import torch.nn as nn


class Asin(nn.Module):

    def forward(self, x):
        return torch.asin(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
