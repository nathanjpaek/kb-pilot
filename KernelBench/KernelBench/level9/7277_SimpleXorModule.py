import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleXorModule(torch.nn.Module):

    def __init__(self):
        super(SimpleXorModule, self).__init__()

    def forward(self, a, b):
        c = torch.logical_xor(a, b)
        return torch.logical_xor(c, c)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
