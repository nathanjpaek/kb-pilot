import torch
import torch.nn.functional as F
import torch.jit
import torch.onnx
import torch.nn


class SimpleSoftmaxModel(torch.nn.Module):

    def __init__(self, dimension):
        super(SimpleSoftmaxModel, self).__init__()
        self.dimension = dimension

    def forward(self, tensor):
        return F.softmax(tensor, self.dimension)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dimension': 4}]
