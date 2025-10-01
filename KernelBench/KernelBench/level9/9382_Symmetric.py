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


class Symmetric(nn.Module):

    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
