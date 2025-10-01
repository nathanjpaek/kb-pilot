import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralBlock(nn.Module):

    def __init__(self, c_planes, p_planes, out_planes):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes, p_planes, kernel_size=1, padding
            =0, stride=1)
        self.top = nn.Conv2d(p_planes, out_planes, kernel_size=3, padding=1,
            stride=1)

    def forward(self, c, p):
        _, _, H, W = c.size()
        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2, mode='nearest')
        p = p[:, :, :H, :W] + c
        p = self.top(p)
        return p


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c_planes': 4, 'p_planes': 4, 'out_planes': 4}]
