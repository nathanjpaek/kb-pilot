import torch
import torch.nn as nn


class Upsample(nn.Module):

    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, self.stride, W, self.
            stride).contiguous().view(B, C, H * self.stride, W * self.stride)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
