import torch
import torch.nn as nn
import torch.nn.functional as F


class Smooth_loss(nn.Module):

    def __init__(self, Smooth_weight=1):
        super(Smooth_loss, self).__init__()
        self.Smooth_weight = Smooth_weight

    def forward(self, x):
        _b, _c, h, w = x.size()
        x_h = F.pad(x, (0, 0, 1, 1))
        h_tv = torch.mean(torch.pow(x_h[:, :, 2:, :] - x_h[:, :, :h, :], 2))
        x_w = F.pad(x, (1, 1, 0, 0))
        w_tv = torch.mean(torch.pow(x_w[:, :, :, 2:] - x_w[:, :, :, :w], 2))
        self.loss = (h_tv + w_tv) / 2
        return self.loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
