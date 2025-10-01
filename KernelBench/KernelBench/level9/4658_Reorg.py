import torch
import torch.nn as nn
import torch.utils.data


class Reorg(nn.Module):

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert H % stride == 0
        assert W % stride == 0
        ws = stride
        hs = stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4
            ).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3
            ).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2
            ).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
