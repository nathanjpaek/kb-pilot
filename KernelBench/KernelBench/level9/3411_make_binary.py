import torch
from torch import Tensor


class make_binary(torch.nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, tensor: 'Tensor') ->Tensor:
        return tensor % 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
