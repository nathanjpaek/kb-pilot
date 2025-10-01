import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleExpModule(torch.nn.Module):

    def forward(self, input):
        other = torch.exp(input)
        return torch.exp(other)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
