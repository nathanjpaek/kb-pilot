import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleSumModule(torch.nn.Module):

    def __init__(self, dtype=None):
        super(SimpleSumModule, self).__init__()
        self.dtype = dtype

    def forward(self, a):
        b = a + a
        return torch.sum(b, dtype=self.dtype)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
