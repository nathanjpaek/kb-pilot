import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleArgSortModule(torch.nn.Module):

    def __init__(self, descending=True):
        super(SimpleArgSortModule, self).__init__()
        self.descending = descending

    def forward(self, inputs):
        return torch.argsort(inputs, dim=-1, descending=self.descending)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
