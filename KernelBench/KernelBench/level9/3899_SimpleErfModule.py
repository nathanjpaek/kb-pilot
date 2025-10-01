import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleErfModule(torch.nn.Module):

    def forward(self, input):
        return torch.special.erf(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
