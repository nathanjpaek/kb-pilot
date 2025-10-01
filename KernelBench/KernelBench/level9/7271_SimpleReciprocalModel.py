import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleReciprocalModel(torch.nn.Module):

    def __init__(self, inplace=False):
        super(SimpleReciprocalModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        other = tensor + tensor
        return other.reciprocal_() if self.inplace else torch.reciprocal(other)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
