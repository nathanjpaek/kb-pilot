import torch
import torch.nn as nn
import torch.nn.functional as F


class OneDilate(nn.Module):

    def __init__(self, kernel_size=10, channels=3, gpu=True):
        super(OneDilate, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        gaussian_kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.mean = (self.kernel_size - 1) // 2
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels,
            out_channels=self.channels, kernel_size=self.kernel_size,
            groups=self.channels, bias=False)
        if gpu:
            gaussian_kernel = gaussian_kernel
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        x = F.pad((1 - x) * 0.5, (self.mean, self.mean, self.mean, self.
            mean), 'replicate')
        return self.gaussian_filter(x)


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
