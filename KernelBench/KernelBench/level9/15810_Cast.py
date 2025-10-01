import torch
import torch.onnx
import torch.nn as nn


class Cast(nn.Module):

    def forward(self, x):
        return x.type(torch.int32)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
