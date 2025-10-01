import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from torchvision.models.mobilenetv2 import _make_divisible


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: 'int', squeeze_factor: 'int'=4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: 'Tensor', inplace: 'bool') ->Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: 'Tensor') ->Tensor:
        scale = self._scale(input, True)
        return scale * input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4}]
