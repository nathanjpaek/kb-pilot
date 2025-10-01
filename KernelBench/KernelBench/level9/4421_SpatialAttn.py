import torch
import torch.nn as nn


class SpatialAttn(nn.Module):
    """Spatial Attention Layer"""

    def __init__(self):
        super(SpatialAttn, self).__init__()

    def forward(self, x):
        x = x.mean(1, keepdim=True)
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0), -1)
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0), 1, h, w)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
