import torch
import torch.nn as nn


class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int): stride to divide the input tensor
    """

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        if not isinstance(stride, int):
            raise TypeError(f'stride is not an int [{type(stride)}]')
        self.stride = stride
        self.darknet = True

    def __repr__(self):
        return (
            f'{self.__class__.__name__} (stride={self.stride}, darknet_compatible_mode={self.darknet})'
            )

    def forward(self, x):
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        if H % self.stride != 0:
            raise ValueError(
                f'Dimension mismatch: {H} is not divisible by {self.stride}')
        if W % self.stride != 0:
            raise ValueError(
                f'Dimension mismatch: {W} is not divisible by {self.stride}')
        if self.darknet:
            x = x.view(B, C // self.stride ** 2, H, self.stride, W, self.stride
                ).contiguous()
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
            x = x.view(B, -1, H // self.stride, W // self.stride)
        else:
            ws, hs = self.stride, self.stride
            x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4
                ).contiguous()
            x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3
                ).contiguous()
            x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2
                ).contiguous()
            x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
