import torch
import torch.nn as nn
import torch.nn.functional as F


class SharpenedCosineSimilarity(nn.Conv2d):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size,
        stride=1, padding=None, dilation=1, groups: 'int'=1, bias: 'bool'=
        False, q_init: 'float'=10, p_init: 'float'=1.0, q_scale: 'float'=
        0.3, p_scale: 'float'=5, eps: 'float'=1e-06):
        if padding is None:
            if int(torch.__version__.split('.')[1]) >= 10:
                padding = 'same'
            else:
                padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        bias = False
        assert dilation == 1, 'Dilation has to be 1 to use AvgPool2d as L2-Norm backend.'
        assert groups == in_channels or groups == 1, 'Either depthwise or full convolution. Grouped not supported'
        super(SharpenedCosineSimilarity, self).__init__(in_channels,
            out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.p_scale = p_scale
        self.q_scale = q_scale
        self.p = torch.nn.Parameter(torch.full((out_channels,), float(
            p_init * self.p_scale)))
        self.q = torch.nn.Parameter(torch.full((1,), float(q_init * self.
            q_scale)))
        self.eps = eps

    def forward(self, inp: 'torch.Tensor') ->torch.Tensor:
        out = inp.square()
        if self.groups == 1:
            out = out.sum(1, keepdim=True)
        q = torch.exp(-self.q / self.q_scale)
        norm = F.conv2d(out, torch.ones_like(self.weight[:1, :1]), None,
            self.stride, self.padding, self.dilation)
        norm = norm + (q + self.eps)
        weight = self.weight / self.weight.square().sum(dim=(1, 2, 3),
            keepdim=True).sqrt()
        out = F.conv2d(inp, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups) / norm.sqrt()
        magnitude = out.abs() + self.eps
        sign = out.sign()
        p = torch.exp(self.p / self.p_scale)
        out = magnitude.pow(p.view(1, -1, 1, 1))
        out = out * sign
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
