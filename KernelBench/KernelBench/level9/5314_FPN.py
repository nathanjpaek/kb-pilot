import torch
import torch.nn as nn
from math import sqrt as sqrt
from itertools import product as product
import torch.nn.functional as F


class FPN(nn.Module):

    def __init__(self, lat_inC, top_inC, outC, mode='nearest'):
        super(FPN, self).__init__()
        assert mode in ['nearest', 'bilinear']
        self.latlayer = nn.Conv2d(lat_inC, outC, 1, 1, padding=0)
        self.toplayer = nn.Conv2d(top_inC, outC, 1, 1, padding=0)
        self.up_mode = mode
        self.bottom_smooth = nn.Conv2d(outC, outC, 3, 1, padding=1)

    def forward(self, bottom, top):
        x = self.latlayer(bottom)
        y = self.toplayer(top)
        up_add = F.upsample(y, scale_factor=2, mode=self.up_mode) + x
        out = self.bottom_smooth(up_add)
        return out


def get_inputs():
    return [torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lat_inC': 4, 'top_inC': 4, 'outC': 4}]
