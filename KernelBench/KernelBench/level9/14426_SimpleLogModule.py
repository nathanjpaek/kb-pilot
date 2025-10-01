import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleLogModule(torch.nn.Module):

    def __init__(self, *dimensions):
        super(SimpleLogModule, self).__init__()

    def forward(self, a):
        b = torch.log(a)
        return torch.log(b)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
