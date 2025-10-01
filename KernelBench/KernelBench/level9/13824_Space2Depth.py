import torch
import torch.nn as nn
import torch.optim
import torch._utils
import torch.nn


class Space2Depth(nn.Module):

    def __init__(self, block_size):
        super(Space2Depth, self).__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * self.bs ** 2, H // self.bs, W // self.bs)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'block_size': 4}]
