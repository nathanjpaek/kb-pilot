import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleMaxModule(torch.nn.Module):

    def __init__(self):
        super(SimpleMaxModule, self).__init__()

    def forward(self, a, b):
        return torch.max(a + a, b + b)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
