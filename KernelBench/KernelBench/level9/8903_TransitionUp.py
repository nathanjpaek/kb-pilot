import torch
import torch.nn
import torch.nn.functional as F
import torch.nn as nn


class TransitionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x, skip, concat=True):
        out = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode=
            'bilinear', align_corners=True)
        if concat:
            out = torch.cat([out, skip], 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
