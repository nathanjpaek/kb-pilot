from torch.nn import Module
import torch
from torch import Tensor
from torch.nn import Identity
from torch.nn.modules import Module
import torch.optim.lr_scheduler


class AbsLayer(Module):

    def forward(self, x: 'Tensor') ->Tensor:
        return torch.abs(x).reshape((-1, 1))


class AbsModel(Module):
    """Fake model, that simply compute the absolute value of the inputs"""

    def __init__(self):
        super().__init__()
        self.features = AbsLayer()
        self.classifier = Identity()

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
