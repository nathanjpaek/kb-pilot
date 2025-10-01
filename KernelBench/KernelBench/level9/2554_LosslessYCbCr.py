import torch
import torch.nn.parallel
import torch.utils.data
from torch import nn
import torch.fft


class LosslessYCbCr(nn.Module):

    def forward(self, rgb: 'torch.Tensor'):
        return torch.cat([(rgb[:, 0:1] + 2 * rgb[:, 1:2] + rgb[:, 2:3]) / 4,
            rgb[:, 2:3] - rgb[:, 1:2], rgb[:, 0:1] - rgb[:, 1:2]], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
