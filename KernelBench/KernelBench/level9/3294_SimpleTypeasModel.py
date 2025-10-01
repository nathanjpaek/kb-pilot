import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleTypeasModel(torch.nn.Module):

    def __init__(self):
        super(SimpleTypeasModel, self).__init__()

    def forward(self, tensor, other=None):
        other = tensor if other is None else other
        if tensor.dtype != torch.bool:
            tensor = tensor + tensor
        typed = tensor.type_as(other)
        return typed + typed


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
