import torch
import torch.nn as nn


class MinibatchStdLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, group_size=4):
        group_size = min(group_size, x.shape[0])
        _channels, height, width = x.shape[1:]
        y = x.view(group_size, -1, *x.shape[1:])
        y = y.float()
        y -= y.mean(dim=0, keepdim=True)
        y = y.pow(2).mean(dim=0)
        y = (y + 1e-08).sqrt()
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(group_size, 1, height, width)
        return torch.cat((x, y), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
