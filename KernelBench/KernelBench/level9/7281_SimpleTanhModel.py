import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleTanhModel(torch.nn.Module):

    def __init__(self, inplace=False):
        super(SimpleTanhModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        tensor = tensor + tensor
        return tensor.tanh_() if self.inplace else tensor.tanh()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
