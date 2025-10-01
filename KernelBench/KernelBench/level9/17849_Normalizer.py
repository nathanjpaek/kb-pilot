import torch
from torch import nn


class Normalizer(nn.Module):

    def __init__(self, target_norm=1.0):
        super().__init__()
        self.target_norm = target_norm

    def forward(self, input: 'torch.Tensor'):
        return input * self.target_norm / input.norm(p=2, dim=1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
