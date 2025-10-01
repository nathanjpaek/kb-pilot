import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleCumSumModule(torch.nn.Module):

    def __init__(self, dim):
        super(SimpleCumSumModule, self).__init__()
        self.dim = dim

    def forward(self, tensor):
        return torch.cumsum(tensor, self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
