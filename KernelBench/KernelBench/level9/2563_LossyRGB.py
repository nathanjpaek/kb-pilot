import torch
import torch.nn.parallel
import torch.utils.data
from torch import nn
import torch.fft


class LossyRGB(nn.Module):

    def forward(self, ycbcr: 'torch.Tensor'):
        return torch.cat([ycbcr[:, 0:1] + 1.402 * ycbcr[:, 2:3], ycbcr[:, 0
            :1] - 0.34413 * ycbcr[:, 1:2] - 0.71414 * ycbcr[:, 2:3], ycbcr[
            :, 0:1] + 1.772 * ycbcr[:, 1:2]], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
