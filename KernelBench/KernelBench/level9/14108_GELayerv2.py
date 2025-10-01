import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed


class GELayerv2(nn.Module):

    def __init__(self):
        super(GELayerv2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        _b, _c, _, _ = x.size()
        res = x
        y = self.avg_pool(x)
        y = self.sigmod(y)
        z = x * y
        return res + z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
