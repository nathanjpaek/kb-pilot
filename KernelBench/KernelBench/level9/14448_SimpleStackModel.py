import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleStackModel(torch.nn.Module):

    def __init__(self, dim):
        super(SimpleStackModel, self).__init__()
        self.dim = dim

    def forward(self, a, b):
        c = b + b
        return torch.stack((a, c), dim=self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
