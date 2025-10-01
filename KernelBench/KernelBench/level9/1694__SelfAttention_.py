import torch
from torch import nn


class _SelfAttention_(nn.Module):

    def __init__(self, in_planes):
        super(_SelfAttention_, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, x, mask=None):
        F = self.f(x)
        G = self.g(x)
        H = self.h(x)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        if mask is not None:
            mask = mask.view(b, -1, w * h)
            mask = torch.bmm(mask.permute(0, 2, 1), mask)
            S = S.masked_fill(mask == 0, -1000000000.0)
        S = self.sm(S)
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += x
        return O


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4}]
