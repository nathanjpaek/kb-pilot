import torch
from torch import nn
import torch.utils.data


class ReOrgLayer(nn.Module):

    def __init__(self, stride=2):
        super(ReOrgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert x.data.dim() == 4
        B, C, H, W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert H % hs == 0, 'The stride ' + str(self.stride
            ) + ' is not a proper divisor of height ' + str(H)
        assert W % ws == 0, 'The stride ' + str(self.stride
            ) + ' is not a proper divisor of height ' + str(W)
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(-2, -3
            ).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs, ws)
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(-1, -2
            ).contiguous()
        x = x.view(B, C, ws * hs, H // ws, W // ws).transpose(1, 2).contiguous(
            )
        x = x.view(B, C * ws * hs, H // ws, W // ws)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
