import torch
import torch.nn as nn


class ConvLR(nn.Module):
    """[u * v + res] version of torch.nn.ConvLR"""

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,
        rank_ratio=0.25, bias=True, device=None, dtype=None):
        super().__init__()
        sliced_rank = int(min(in_planes, out_planes) * rank_ratio)
        self.u = nn.Conv2d(in_channels=in_planes, out_channels=sliced_rank,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=
            False, device=device, dtype=dtype)
        self.v = nn.Conv2d(in_channels=sliced_rank, out_channels=out_planes,
            kernel_size=1, stride=1, padding=0, bias=bias, device=device,
            dtype=dtype)
        self.res = nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=
            False, device=device, dtype=dtype)

    def forward(self, input):
        return self.v(self.u(input)) + self.res(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
