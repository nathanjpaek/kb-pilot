import torch
import torch.utils.data
import torch
from torch import nn


class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""

    def __init__(self, epsilon=1e-08):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(
                f'The input tensor should be with shape [batch_size, channel, height, width], but {x.shape} received!'
                )
        x = x - torch.mean(x, dim=[2, 3], keepdim=True)
        x = x / torch.sqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) +
            self.eps)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
