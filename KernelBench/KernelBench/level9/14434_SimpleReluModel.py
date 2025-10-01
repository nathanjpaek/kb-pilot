import torch
import torch.jit
import torch.nn.functional as F
import torch.onnx
import torch.nn


class SimpleReluModel(torch.nn.Module):

    def __init__(self, inplace=False):
        super(SimpleReluModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        other = F.relu(tensor, inplace=self.inplace)
        return F.relu(other, inplace=self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
