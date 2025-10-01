import torch
import torch.onnx
import torch.nn as nn


class ReduceSum3(nn.Module):

    def forward(self, x):
        return torch.sum(x, (1, 3), keepdim=False)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
