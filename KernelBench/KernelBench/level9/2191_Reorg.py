import torch
import torch.nn as nn


class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int): stride to divide the input tensor
    """

    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
        if not isinstance(stride, int):
            raise TypeError(f'stride is not an int [{type(stride)}]')

    def extra_repr(self):
        return f'stride={self.stride}'

    def forward(self, x):
        assert x.dim() == 4
        B, C, H, W = x.size()
        if H % self.stride != 0:
            raise ValueError(
                f'Dimension mismatch: {H} is not divisible by {self.stride}')
        if W % self.stride != 0:
            raise ValueError(
                f'Dimension mismatch: {W} is not divisible by {self.stride}')
        x = x.view(B, C // self.stride ** 2, H, self.stride, W, self.stride
            ).contiguous()
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, -1, H // self.stride, W // self.stride)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
