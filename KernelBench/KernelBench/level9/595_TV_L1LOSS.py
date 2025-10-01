import torch
import torch.nn as nn
import torch.utils.data


class TV_L1LOSS(nn.Module):

    def __init__(self):
        super(TV_L1LOSS, self).__init__()

    def forward(self, x, y):
        size = x.size()
        h_tv_diff = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :] - (y[:, :, 1
            :, :] - y[:, :, :-1, :])).sum()
        w_tv_diff = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1] - (y[:, :, :,
            1:] - y[:, :, :, :-1])).sum()
        return (h_tv_diff + w_tv_diff) / size[0] / size[1] / size[2] / size[3]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
