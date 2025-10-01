import torch
import torch.nn as nn


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        h_x, w_x = x.size()[2:]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1])
        loss = torch.sum(h_tv) + torch.sum(w_tv)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
