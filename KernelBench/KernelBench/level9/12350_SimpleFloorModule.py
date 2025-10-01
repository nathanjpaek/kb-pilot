import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleFloorModule(torch.nn.Module):

    def forward(self, a, b):
        c = a + b
        return torch.floor(c)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
