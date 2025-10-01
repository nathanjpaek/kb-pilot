import torch
import torch.nn as nn
import torch.onnx


class Scale(torch.nn.Module):

    def __init__(self, value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, input):
        return self.scale * input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
