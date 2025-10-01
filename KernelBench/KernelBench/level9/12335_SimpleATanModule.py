import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleATanModule(torch.nn.Module):

    def __init__(self):
        super(SimpleATanModule, self).__init__()

    def forward(self, a):
        return torch.atan(a + a)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
