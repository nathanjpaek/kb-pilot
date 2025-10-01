import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class sobel_net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_opx = nn.Conv2d(1, 1, 3, bias=False)
        self.conv_opy = nn.Conv2d(1, 1, 3, bias=False)
        sobel_kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype='float32').reshape((1, 1, 3, 3))
        sobel_kernely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype='float32').reshape((1, 1, 3, 3))
        self.conv_opx.weight.data = torch.from_numpy(sobel_kernelx)
        self.conv_opy.weight.data = torch.from_numpy(sobel_kernely)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, im):
        x = (0.299 * im[:, 0, :, :] + 0.587 * im[:, 1, :, :] + 0.114 * im[:,
            2, :, :]).unsqueeze(1)
        gradx = self.conv_opx(x)
        grady = self.conv_opy(x)
        x = (gradx ** 2 + grady ** 2) ** 0.5
        x = (x - x.min()) / (x.max() - x.min())
        x = F.pad(x, (1, 1, 1, 1))
        x = torch.cat([im, x], dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
