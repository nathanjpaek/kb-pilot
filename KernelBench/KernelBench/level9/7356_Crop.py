import torch
from typing import cast
from torch import nn
from torchvision.transforms import functional as F
import torch.nn.functional as F
import torchvision.transforms.functional as F
import torch.autograd


class Crop(nn.Module):

    def __init__(self, *, top: int, left: int, height: int, width: int) ->None:
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, image: 'torch.Tensor') ->torch.Tensor:
        return cast(torch.Tensor, F.crop(image, top=self.top, left=self.
            left, height=self.height, width=self.width))

    def extra_repr(self) ->str:
        return ', '.join([f'top={self.top}', f'left={self.left}',
            f'height={self.height}', f'width={self.width}'])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'top': 4, 'left': 4, 'height': 4, 'width': 4}]
