import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.cuda
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class LowRankConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=True, rank=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size
            , int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int
            ) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int
            ) else dilation
        self.rank = rank
        self.G = nn.Parameter(torch.Tensor(self.kernel_size[0] * self.
            kernel_size[1], self.rank, self.in_channels))
        self.H = nn.Parameter(torch.Tensor(self.kernel_size[0] * self.
            kernel_size[1], self.out_channels, self.rank))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in, _fan_out = self.in_channels, self.out_channels
        nn.init.uniform_(self.G, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
        nn.init.uniform_(self.H, -1 / math.sqrt(self.rank), 1 / math.sqrt(
            self.rank))
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        M = torch.bmm(self.H, self.G).permute(1, 2, 0).reshape(self.
            out_channels, self.in_channels, *self.kernel_size)
        return F.conv2d(x, M, self.bias, self.stride, self.padding, self.
            dilation)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
