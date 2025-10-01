import torch
import torch.onnx
import torch.nn as nn


class Permute(nn.Module):

    def forward(self, x):
        x = x + 1.0
        return x.permute(2, 0, 1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
