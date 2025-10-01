import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleReshapeModel(torch.nn.Module):

    def __init__(self, shape):
        super(SimpleReshapeModel, self).__init__()
        self.shape = shape

    def forward(self, tensor):
        combined = tensor + tensor
        return combined.reshape(self.shape)


def get_inputs():
    return [torch.rand([4])]


def get_init_inputs():
    return [[], {'shape': 4}]
