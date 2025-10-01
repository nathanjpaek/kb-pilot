import torch
import torch.utils.data
import torch
from torch import nn


class GaussianFilter(nn.Module):

    def __init__(self, kernel_size=13, stride=1, padding=6):
        super(GaussianFilter, self).__init__()
        mean = (kernel_size - 1) / 2.0
        variance = ((kernel_size - 1) / 6.0) ** 2.0
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2.0, dim
            =-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride,
            padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
