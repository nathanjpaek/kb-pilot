import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class FixedBlurLayer(nn.Module):

    def __init__(self, kernel):
        super(FixedBlurLayer, self).__init__()
        self.kernel = kernel
        to_pad_x = int((self.kernel.shape[0] - 1) / 2)
        to_pad_y = int((self.kernel.shape[1] - 1) / 2)
        self.pad = nn.ReflectionPad2d((to_pad_x, to_pad_x, to_pad_y, to_pad_y))
        self.mask_np = np.zeros(shape=(1, 3, self.kernel.shape[0], self.
            kernel.shape[1]))
        self.mask_np[0, 0, :, :] = self.kernel
        self.mask_np[0, 1, :, :] = self.kernel
        self.mask_np[0, 2, :, :] = self.kernel
        self.mask = nn.Parameter(data=torch.FloatTensor(self.mask_np),
            requires_grad=False)

    def forward(self, x):
        return F.conv2d(self.pad(x), self.mask)


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {'kernel': torch.rand([4, 4])}]
