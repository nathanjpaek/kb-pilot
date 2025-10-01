import torch
import torch.nn as nn


class Mask_BN(nn.Module):

    def __init__(self):
        super(Mask_BN, self).__init__()

    def forward(self, x):
        x_mask = x != 0
        x_centralization = x - x_mask * x[:, 0, :, :].unsqueeze(1)
        none_zero_n = x_mask.sum(axis=3).sum(axis=2).sum(axis=1).unsqueeze(1)
        none_zero_sum = x_centralization.sum(axis=2).sum(axis=1)
        x_mean = none_zero_sum / (none_zero_n * 0.5)
        mu = x_mean.unsqueeze(1).unsqueeze(2) * x_mask
        var = (((x_centralization - mu) ** 2).sum(axis=2).sum(axis=1) /
            none_zero_n).unsqueeze(1).unsqueeze(2)
        bn_x = (x_centralization - mu) / var ** 0.5
        return bn_x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
