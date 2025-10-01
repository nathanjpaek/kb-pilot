import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleAbsModule(torch.nn.Module):

    def __init__(self):
        super(SimpleAbsModule, self).__init__()

    def forward(self, a):
        return torch.abs(a + a)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
