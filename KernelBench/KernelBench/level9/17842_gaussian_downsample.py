import math
import torch
import torch.nn as nn


class gaussian_downsample(nn.Module):
    """
    Downsampling module with Gaussian filtering
    """

    def __init__(self, kernel_size, sigma, stride, pad=False):
        super(gaussian_downsample, self).__init__()
        self.gauss = nn.Conv2d(3, 3, kernel_size, stride=stride, groups=3,
            bias=False)
        gaussian_weights = self.init_weights(kernel_size, sigma)
        self.gauss.weight.data = gaussian_weights
        self.gauss.weight.requires_grad_(False)
        self.pad = pad
        self.padsize = kernel_size - 1

    def forward(self, x):
        if self.pad:
            x = torch.cat((x, x[:, :, :self.padsize, :]), 2)
            x = torch.cat((x, x[:, :, :, :self.padsize]), 3)
        return self.gauss(x)

    def init_weights(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.0
        gaussian_kernel = 1.0 / (2.0 * math.pi * variance) * torch.exp(-
            torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(
            3, 1, 1, 1)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'kernel_size': 4, 'sigma': 4, 'stride': 1}]
