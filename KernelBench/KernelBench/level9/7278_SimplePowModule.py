import torch
import torch.jit
import torch.onnx
import torch.nn


class SimplePowModule(torch.nn.Module):

    def __init__(self, power):
        super(SimplePowModule, self).__init__()
        self.power = power

    def forward(self, tensor):
        return torch.pow(tensor, self.power)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'power': 4}]
