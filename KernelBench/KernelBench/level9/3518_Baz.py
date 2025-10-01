import torch
import torch.onnx
import torch.nn


class Baz(torch.nn.Module):

    def __init__(self, x):
        super(Baz, self).__init__()
        self.x = x

    def forward(self, a, b):
        return a + b * self.x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'x': 4}]
