import torch
from torch import Tensor
from typing import Optional
from torch import nn
from torch.nn import functional as F


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


class SqueezeExcitation(nn.Module):

    def __init__(self, ch, squeeze_factor=4):
        super().__init__()
        squeeze_ch = _make_divisible(ch // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(ch, squeeze_ch, 1)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Conv2d(squeeze_ch, ch, 1)

    def _scale(self, x: 'Tensor') ->Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc2(self.relu(self.fc1(scale)))
        return F.hardsigmoid(scale, True)

    def forward(self, x: 'Tensor') ->Tensor:
        scale = self._scale(x)
        return scale * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ch': 4}]
