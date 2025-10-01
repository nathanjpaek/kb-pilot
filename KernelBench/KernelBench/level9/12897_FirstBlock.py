import torch
import numpy as np
import torch.nn as nn


class BatchNormLayer(nn.Module):
    """Implements batch normalization layer."""

    def __init__(self, channels, gamma=False, beta=True, decay=0.9, epsilon
        =1e-05):
        """Initializes with basic settings.

    Args:
      channels: Number of channels of the input tensor.
      gamma: Whether the scale (weight) of the affine mapping is learnable.
      beta: Whether the center (bias) of the affine mapping is learnable.
      decay: Decay factor for moving average operations in this layer.
      epsilon: A value added to the denominator for numerical stability.
    """
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=channels, affine=True,
            track_running_stats=True, momentum=1 - decay, eps=epsilon)
        self.bn.weight.requires_grad = gamma
        self.bn.bias.requires_grad = beta

    def forward(self, x):
        return self.bn(x)


class FirstBlock(nn.Module):
    """Implements the first block, which is a convolutional block."""

    def __init__(self, in_channels, out_channels, use_wscale=False,
        wscale_gain=np.sqrt(2.0), use_bn=False, activation_type='lrelu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.scale = wscale_gain / np.sqrt(in_channels * 3 * 3
            ) if use_wscale else 1.0
        self.bn = BatchNormLayer(channels=out_channels
            ) if use_bn else nn.Identity()
        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(
                f'Not implemented activation function: {activation_type}!')

    def forward(self, x):
        return self.activate(self.bn(self.conv(x) * self.scale))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
