import abc
import torch
import torch.nn
import torch.optim


class MapIDList(torch.nn.Module):

    @abc.abstractmethod
    def forward(self, raw_values: 'torch.Tensor') ->torch.Tensor:
        pass


class ModuloMapIDList(MapIDList):

    def __init__(self, modulo: 'int'):
        super().__init__()
        self.modulo = modulo

    def forward(self, raw_values: 'torch.Tensor') ->torch.Tensor:
        return torch.remainder(raw_values, self.modulo)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'modulo': 4}]
