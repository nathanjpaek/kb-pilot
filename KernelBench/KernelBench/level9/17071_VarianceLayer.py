import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class VarianceLayer(nn.Module):

    def __init__(self, patch_size=5, channels=1):
        self.patch_size = patch_size
        super(VarianceLayer, self).__init__()
        mean_mask = np.ones((channels, channels, patch_size, patch_size)) / (
            patch_size * patch_size)
        self.mean_mask = nn.Parameter(data=torch.FloatTensor(mean_mask),
            requires_grad=False)
        mask = np.zeros((channels, channels, patch_size, patch_size))
        mask[:, :, patch_size // 2, patch_size // 2] = 1.0
        self.ones_mask = nn.Parameter(data=torch.FloatTensor(mask),
            requires_grad=False)

    def forward(self, x):
        Ex_E = F.conv2d(x, self.ones_mask) - F.conv2d(x, self.mean_mask)
        return F.conv2d(Ex_E ** 2, self.mean_mask)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
