import torch
from typing import List
from typing import Union
import torch.nn as nn


def autopad(kernel_size: 'Union[int, List[int]]', padding:
    'Union[int, None]'=None) ->Union[int, List[int]]:
    """Auto padding calculation for pad='same' in TensorFlow."""
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]
    return padding or [(x // 2) for x in kernel_size]


class Activation:
    """Convert string activation name to the activation class."""

    def __init__(self, act_type: 'Union[str, None]') ->None:
        """Convert string activation name to the activation class.

        Args:
            type: Activation name.

        Returns:
            nn.Identity if {type} is None.
        """
        self.type = act_type
        self.args = [1] if self.type == 'Softmax' else []

    def __call__(self) ->nn.Module:
        if self.type is None:
            return nn.Identity()
        elif hasattr(nn, self.type):
            return getattr(nn, self.type)(*self.args)
        else:
            return getattr(__import__('src.modules.activations', fromlist=[
                '']), self.type)()


class CR(nn.Module):
    """Standard convolution with batch normalization and activation."""

    def __init__(self, in_channel: 'int', out_channels: 'int', kernel_size:
        'int', stride: 'int'=1, padding: 'Union[int, None]'=None, groups:
        'int'=1, activation: 'Union[str, None]'='ReLU') ->None:
        """Standard convolution with batch normalization and activation.

        Args:
            in_channel: input channels.
            out_channels: output channels.
            kernel_size: kernel size.
            stride: stride.
            padding: input padding. If None is given, autopad is applied
                which is identical to padding='SAME' in TensorFlow.
            groups: group convolution.
            activation: activation name. If None is given, nn.Identity is applied
                which is no activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channels, kernel_size, stride,
            padding=autopad(kernel_size, padding), groups=groups, bias=True)
        self.act = Activation(activation)()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward."""
        return self.act(self.conv(x))

    def fusefoward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Fuse forward."""
        return self.act(self.conv(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channels': 4, 'kernel_size': 4}]
