import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.nn


class HessianResp(nn.Module):

    def __init__(self):
        super(HessianResp, self).__init__()
        self.gx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]
            ], dtype=np.float32))
        self.gy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-
            0.5]]]], dtype=np.float32))
        self.gxx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gxx.weight.data = torch.from_numpy(np.array([[[[1.0, -2.0, 1.0
            ]]]], dtype=np.float32))
        self.gyy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gyy.weight.data = torch.from_numpy(np.array([[[[1.0], [-2.0],
            [1.0]]]], dtype=np.float32))
        return

    def forward(self, x, scale):
        gxx = self.gxx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gyy = self.gyy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        gxy = self.gy(F.pad(self.gx(F.pad(x, (1, 1, 0, 0), 'replicate')), (
            0, 0, 1, 1), 'replicate'))
        return torch.abs(gxx * gyy - gxy * gxy) * scale ** 4


def get_inputs():
    return [torch.rand([4, 1, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
