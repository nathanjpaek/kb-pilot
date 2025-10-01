import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.nn


class LocalNorm2d(nn.Module):

    def __init__(self, kernel_size=33):
        super(LocalNorm2d, self).__init__()
        self.ks = kernel_size
        self.pool = nn.AvgPool2d(kernel_size=self.ks, stride=1, padding=0)
        self.eps = 1e-10
        return

    def forward(self, x):
        pd = int(self.ks / 2)
        mean = self.pool(F.pad(x, (pd, pd, pd, pd), 'reflect'))
        return torch.clamp((x - mean) / (torch.sqrt(torch.abs(self.pool(F.
            pad(x * x, (pd, pd, pd, pd), 'reflect')) - mean * mean)) + self
            .eps), min=-6.0, max=6.0)


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {}]
