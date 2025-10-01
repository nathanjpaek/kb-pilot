import torch
import torch.onnx
import torch
import torch.nn as nn
import torch.nn.functional as F


class hswish(nn.Module):

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
