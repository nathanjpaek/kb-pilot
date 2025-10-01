import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyDilateBlur(nn.Module):

    def __init__(self, kernel_size=7, channels=3, sigma=0.8):
        super(MyDilateBlur, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        x_cord = torch.arange(self.kernel_size + 0.0)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size,
            self.kernel_size)
        y_grid = x_grid.t()
        self.xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.mean = (self.kernel_size - 1) // 2
        self.diff = -torch.sum((self.xy_grid - self.mean) ** 2.0, dim=-1)
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels,
            out_channels=self.channels, kernel_size=self.kernel_size,
            groups=self.channels, bias=False)
        self.gaussian_filter.weight.requires_grad = False
        variance = sigma ** 2.0
        gaussian_kernel = 1.0 / (2.0 * math.pi * variance) * torch.exp(self
            .diff / (2 * variance))
        gaussian_kernel = 2 * gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self
            .kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.gaussian_filter.weight.data = gaussian_kernel

    def forward(self, x):
        y = self.gaussian_filter(F.pad(1 - x, (self.mean, self.mean, self.
            mean, self.mean), 'replicate'))
        return 1 - 2 * torch.clamp(y, min=0, max=1)


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
