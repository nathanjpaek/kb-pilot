import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialShift2d(nn.Module):

    def __init__(self, channels, padding_mode='replicate'):
        super(SpatialShift2d, self).__init__()
        qc = channels // 4
        self.num_shift_left = qc
        self.num_shift_right = qc
        self.num_shift_up = qc
        self.num_shift_down = channels - qc * 3
        self.padding_mode = padding_mode

    def forward(self, x):
        _l, _r, _u, _d = (self.num_shift_left, self.num_shift_right, self.
            num_shift_up, self.num_shift_down)
        x = F.pad(x, (1, 1, 1, 1), self.padding_mode)
        l, r, u, d = torch.split(x, [_l, _r, _u, _d], dim=1)
        l = l[:, :, 1:-1, 0:-2]
        r = r[:, :, 1:-1, 2:]
        u = u[:, :, 0:-2, 1:-1]
        d = d[:, :, 2:, 1:-1]
        x = torch.cat([l, r, u, d], dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
