import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleClampModel(torch.nn.Module):

    def __init__(self, min, max):
        super(SimpleClampModel, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return torch.clamp(input, self.min, self.max)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'min': 4, 'max': 4}]
