import torch
import torch.nn as nn
import torch._utils
import torch.optim


class ChannelNorm(nn.Module):

    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, _h, _w = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
