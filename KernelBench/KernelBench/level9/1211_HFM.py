import torch
import torch.nn as nn


class HFM(nn.Module):

    def __init__(self, k=2):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.AvgPool2d(kernel_size=self.k, stride=
            self.k), nn.Upsample(scale_factor=self.k, mode='nearest'))

    def forward(self, tL):
        assert tL.shape[2] % self.k == 0, 'h, w must divisible by k'
        return tL - self.net(tL)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
