import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleClampMinModel(torch.nn.Module):

    def __init__(self, min):
        super(SimpleClampMinModel, self).__init__()
        self.min = min

    def forward(self, input):
        return torch.clamp_min(input, self.min)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'min': 4}]
