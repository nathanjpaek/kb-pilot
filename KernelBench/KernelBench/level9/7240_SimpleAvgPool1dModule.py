import torch
import torch.nn.functional as F
import torch.jit
import torch.onnx
import torch.nn


class SimpleAvgPool1dModule(torch.nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0):
        super(SimpleAvgPool1dModule, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, inputs):
        return F.avg_pool1d(inputs, self.kernel_size, padding=self.padding,
            stride=self.stride)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
