import torch
from torch import nn
import torch.onnx


class GlobalAveragePool(nn.Module):

    def forward(self, input: 'torch.Tensor'):
        spatial_shape = input.ndimension() - 2
        dim = tuple(range(spatial_shape, spatial_shape + 2))
        return torch.mean(input, dim=dim, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
