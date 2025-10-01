import torch
import torch.nn.functional as F
import torch.jit
import torch.onnx
import torch.nn


class SimpleLogSoftmaxModel(torch.nn.Module):

    def __init__(self, dimension):
        super(SimpleLogSoftmaxModel, self).__init__()
        self.dimension = dimension

    def forward(self, tensor):
        return F.log_softmax(tensor, self.dimension)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dimension': 4}]
