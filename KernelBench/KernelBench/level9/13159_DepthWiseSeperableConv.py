import torch
import torch.nn as nn


class DepthWiseSeperableConv(nn.Module):

    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__()
        if 'groups' in kwargs:
            del kwargs['groups']
        self.depthwise = nn.Conv2d(in_dim, in_dim, *args, groups=in_dim, **
            kwargs)
        self.pointwise = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'kernel_size': 4}]
