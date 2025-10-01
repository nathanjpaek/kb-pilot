import torch
import torch.nn as nn
from torch.nn import functional as F


class AveragePoolingLayer(nn.Module):
    """Implements the average pooling layer.

  Basically, this layer can be used to downsample feature maps from spatial
  domain.
  """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        ksize = [self.scale_factor, self.scale_factor]
        strides = [self.scale_factor, self.scale_factor]
        return F.avg_pool2d(x, kernel_size=ksize, stride=strides, padding=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
