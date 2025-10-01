import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleSinModule(torch.nn.Module):

    def __init__(self):
        super(SimpleSinModule, self).__init__()

    def forward(self, a):
        return torch.sin(a + a)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
