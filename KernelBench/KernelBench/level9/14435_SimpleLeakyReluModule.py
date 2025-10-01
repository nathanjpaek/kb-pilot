import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleLeakyReluModule(torch.nn.Module):

    def __init__(self, negative_slope=0.01, inplace=False):
        super(SimpleLeakyReluModule, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, a):
        return torch.nn.functional.leaky_relu(a, negative_slope=self.
            negative_slope, inplace=self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
