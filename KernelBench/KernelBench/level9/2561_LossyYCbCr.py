import torch
import torch.nn.parallel
import torch.utils.data
from torch import nn
import torch.fft


class LossyYCbCr(nn.Module):

    def forward(self, rgb: 'torch.Tensor'):
        return torch.cat([0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 *
            rgb[:, 2:3], -0.16875 * rgb[:, 0:1] - 0.33126 * rgb[:, 1:2] + 
            0.5 * rgb[:, 2:3], 0.5 * rgb[:, 0:1] - 0.41869 * rgb[:, 1:2] - 
            0.08131 * rgb[:, 2:3]], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
