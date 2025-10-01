import torch
import torch.onnx
import torch.nn


class SimpleSliceModel(torch.nn.Module):

    def __init__(self):
        super(SimpleSliceModel, self).__init__()

    def forward(self, tensor):
        other = (tensor + tensor)[1:]
        return other[0][1:]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
