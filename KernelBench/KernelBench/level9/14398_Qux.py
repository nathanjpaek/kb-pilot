import torch
import torch.jit
import torch.onnx
import torch.nn


class Qux(torch.nn.Module):

    def __init__(self, x):
        super(Qux, self).__init__()
        self.x = x

    def forward(self, a, b):
        return a - b - self.x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'x': 4}]
