import torch
import torch.jit
import torch.onnx
import torch.nn


class DetachModel(torch.nn.Module):

    def __init__(self):
        super(DetachModel, self).__init__()

    def forward(self, a):
        b = a.detach()
        return b + b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
