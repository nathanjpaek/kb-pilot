import torch
import torch.nn as nn
import torch.quantization
import torch.onnx
import torch.nn.parallel
import torch.utils.data
import torch.fx
import torch.nn
import torch.optim
import torch.profiler


class Skew(nn.Module):

    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)

    def right_inverse(self, A):
        return A.triu(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
