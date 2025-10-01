import torch
import torch.nn as nn


class Reorg(nn.Module):

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert H % self.stride == 0
        assert W % self.stride == 0
        w_stride = self.stride
        h_stride = self.stride
        x = x.view(B, C, H // h_stride, h_stride, W // w_stride, w_stride
            ).transpose(3, 4).contiguous()
        x = x.view(B, C, H // h_stride * (W // w_stride), h_stride * w_stride
            ).transpose(2, 3).contiguous()
        x = x.view(B, C, h_stride * w_stride, H // h_stride, W // w_stride
            ).transpose(1, 2).contiguous()
        x = x.view(B, h_stride * w_stride * C, H // h_stride, W // w_stride)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
