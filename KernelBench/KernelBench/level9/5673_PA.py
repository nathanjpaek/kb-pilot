import torch
import torch.nn as nn


class PA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
