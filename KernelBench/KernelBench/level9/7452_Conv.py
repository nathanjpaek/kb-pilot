import torch
from torch import nn
from torch.nn.functional import interpolate
from typing import cast


class Interpolate(nn.Module):

    def __init__(self, scale_factor: 'float'=1.0, mode: 'str'='nearest'
        ) ->None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return cast(torch.Tensor, interpolate(input, scale_factor=self.
            scale_factor, mode=self.mode))

    def extra_repr(self) ->str:
        extras = [f'scale_factor={self.scale_factor}']
        if self.mode != 'nearest':
            extras.append(f'mode={self.mode}')
        return ', '.join(extras)


class Conv(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'int', *, stride: int=1, upsample: bool=False, norm: bool=True,
        activation: bool=True):
        super().__init__()
        self.upsample = Interpolate(scale_factor=stride) if upsample else None
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=1 if upsample else stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True
            ) if norm else None
        self.activation = nn.ReLU() if activation else None

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        if self.upsample:
            input = self.upsample(input)
        output = self.conv(self.pad(input))
        if self.norm:
            output = self.norm(output)
        if self.activation:
            output = self.activation(output)
        return cast(torch.Tensor, output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
