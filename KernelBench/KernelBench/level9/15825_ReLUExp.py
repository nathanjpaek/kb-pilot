import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F


class ReLUExp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(F.relu(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
