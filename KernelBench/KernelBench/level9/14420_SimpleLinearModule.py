import torch
import torch.jit
import torch.nn.functional as F
import torch.onnx
import torch.nn


class SimpleLinearModule(torch.nn.Module):

    def __init__(self):
        super(SimpleLinearModule, self).__init__()

    def forward(self, input, weight, bias=None):
        return F.linear(input + input, weight, bias)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
