import torch
import torch.nn as nn


class A2Block(nn.Module):
    """
        Implementation of A2Block(NIPS 2018)
    """

    def __init__(self, inplane, plane):
        super(A2Block, self).__init__()
        self.down = nn.Conv2d(inplane, plane, 1)
        self.up = nn.Conv2d(plane, inplane, 1)
        self.gather_down = nn.Conv2d(inplane, plane, 1)
        self.distribue_down = nn.Conv2d(inplane, plane, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        res = x
        A = self.down(res)
        B = self.gather_down(res)
        b, c, h, _w = A.size()
        A = A.view(b, c, -1)
        B = B.view(b, c, -1)
        B = self.softmax(B)
        B = B.permute(0, 2, 1)
        G = torch.bmm(A, B)
        C = self.distribue_down(res)
        C = C.view(b, c, -1)
        C = self.softmax(C)
        C = C.permute(0, 2, 1)
        atten = torch.bmm(C, G)
        atten = atten.permute(0, 2, 1).view(b, c, h, -1)
        atten = self.up(atten)
        out = res + atten
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplane': 4, 'plane': 4}]
