import torch
import torch.jit
import torch.onnx
import torch.nn


class OneTupleModule(torch.nn.Module):

    def __init__(self):
        super(OneTupleModule, self).__init__()

    def forward(self, x):
        y = 2 * x
        return y,


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
