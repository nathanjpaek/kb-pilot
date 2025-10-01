import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed


class GELayerv1(nn.Module):

    def __init__(self):
        super(GELayerv1, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(15, 15), stride=8)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        _b, _c, h, w = x.size()
        res = x
        y = self.avg_pool(x)
        y = F.upsample(y, size=(h, w), mode='bilinear', align_corners=True)
        y = y * x
        return res + y


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {}]
