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


class LastBlock(nn.Module):
    """Implements the last block, which is a dense block."""

    def __init__(self, in_channels, out_channels, use_wscale=False,
        wscale_gain=1.0, use_bn=False):
        super().__init__()
        self.fc = nn.Linear(in_features=in_channels, out_features=
            out_channels, bias=False)
        self.scale = wscale_gain / np.sqrt(in_channels) if use_wscale else 1.0
        self.bn = BatchNormLayer(channels=out_channels
            ) if use_bn else nn.Identity()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x) * self.scale
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return self.bn(x).view(x.shape[0], x.shape[1])


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
