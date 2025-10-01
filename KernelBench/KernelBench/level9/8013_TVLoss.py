import torch
import torch.utils.data
import torch.nn as nn


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        self._tensor_size(x[:, :, 1:, :])
        self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return h_tv + w_tv

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
