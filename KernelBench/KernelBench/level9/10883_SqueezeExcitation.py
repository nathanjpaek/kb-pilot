import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
from torch import nn


def _make_divisible(v: 'float', divisor: 'int', min_value: 'Optional[int]'=None
    ) ->int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: 'Tensor') ->Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: 'int', squeeze_factor: 'int'=4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv3d(input_channels, squeeze_channels, (1, 1, 1))
        self.swish = swish()
        self.fc2 = nn.Conv3d(squeeze_channels, input_channels, (1, 1, 1))

    def _scale(self, input: 'Tensor') ->Tensor:
        scale = F.adaptive_avg_pool3d(input, 1)
        scale = self.fc1(scale)
        scale = self.swish(scale)
        scale = self.fc2(scale)
        return torch.sigmoid(scale)

    def forward(self, input: 'Tensor') ->Tensor:
        scale = self._scale(input)
        return scale * input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4}]
