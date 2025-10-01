import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class ResolutionScalingLayer(nn.Module):
    """Implements the resolution scaling layer.

  Basically, this layer can be used to upsample feature maps from spatial domain
  with nearest neighbor interpolation.
  """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
