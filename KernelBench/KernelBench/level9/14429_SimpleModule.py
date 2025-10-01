import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleModule(torch.nn.Module):

    def __init__(self):
        super(SimpleModule, self).__init__()

    def forward(self, x):
        y = x + x
        y = y + 2
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
