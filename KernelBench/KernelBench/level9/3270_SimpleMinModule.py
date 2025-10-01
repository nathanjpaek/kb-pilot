import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleMinModule(torch.nn.Module):

    def __init__(self):
        super(SimpleMinModule, self).__init__()

    def forward(self, a, b):
        return torch.min(a + a, b + b)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
