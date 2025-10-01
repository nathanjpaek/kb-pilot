import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleOrModule(torch.nn.Module):

    def __init__(self):
        super(SimpleOrModule, self).__init__()

    def forward(self, a, b):
        c = torch.logical_or(a, b)
        return torch.logical_or(c, c)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
