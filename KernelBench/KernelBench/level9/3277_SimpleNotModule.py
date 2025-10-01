import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleNotModule(torch.nn.Module):

    def __init__(self):
        super(SimpleNotModule, self).__init__()

    def forward(self, a):
        b = torch.logical_not(a)
        return torch.logical_not(b)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
