from torch.nn import Module
import torch
from torch import Tensor
from torch.nn.modules import Module
import torch.optim.lr_scheduler


class AbsLayer(Module):

    def forward(self, x: 'Tensor') ->Tensor:
        return torch.abs(x).reshape((-1, 1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
