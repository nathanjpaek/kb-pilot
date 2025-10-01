import torch
from torch import Tensor


class Normalize(torch.nn.Module):

    def forward(self, x: 'Tensor'):
        return (x - x.mean()) / x.std()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
