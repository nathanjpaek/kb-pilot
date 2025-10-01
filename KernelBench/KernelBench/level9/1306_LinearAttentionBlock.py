import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttentionBlock(nn.Module):

    def __init__(self, in_features):
        super(LinearAttentionBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l + g)
        a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        g = torch.mul(a.expand_as(l), l)
        g = g.view(N, C, -1).sum(dim=2)
        return c.view(N, 1, W, H), g


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
