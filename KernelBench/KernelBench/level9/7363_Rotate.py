import torch
from typing import cast
from torch import nn
from torchvision.transforms import functional as F
import torch.nn.functional as F
import torchvision.transforms.functional as F
import torch.autograd


class Rotate(nn.Module):

    def __init__(self, angle: 'float') ->None:
        super().__init__()
        self.angle = angle

    def forward(self, image: 'torch.Tensor') ->torch.Tensor:
        return cast(torch.Tensor, F.rotate(image, self.angle))

    def extra_repr(self) ->str:
        return f'factor={self.angle}'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'angle': 4}]
