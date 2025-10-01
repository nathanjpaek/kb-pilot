import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in `1`_.

    Concretely, the spatial softmax of each feature map is used to compute a
    weighted mean of the pixel locations, effectively performing a soft arg-max
    over the feature dimension.

    .. _1: https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize: 'bool'=False):
        """Constructor.

        Args:
            normalize: Whether to use normalized image coordinates, i.e.
                coordinates in the range `[-1, 1]`.
        """
        super().__init__()
        self.normalize = normalize

    def _coord_grid(self, h: 'int', w: 'int', device: 'torch.device') ->Tensor:
        if self.normalize:
            return torch.stack(torch.meshgrid(torch.linspace(-1, 1, w,
                device=device), torch.linspace(-1, 1, h, device=device)))
        return torch.stack(torch.meshgrid(torch.arange(0, w, device=device),
            torch.arange(0, h, device=device)))

    def forward(self, x: 'Tensor') ->Tensor:
        assert x.ndim == 4, 'Expecting a tensor of shape (B, C, H, W).'
        _b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)
        xc, yc = self._coord_grid(h, w, x.device)
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
