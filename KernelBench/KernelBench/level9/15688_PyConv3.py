import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
    dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride
        =stride, padding=padding, dilation=dilation, groups=groups, bias=False)


class PyConv3(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7], stride=1,
        pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=
            pyconv_kernels[0], padding=pyconv_kernels[0] // 2, stride=
            stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=
            pyconv_kernels[1], padding=pyconv_kernels[1] // 2, stride=
            stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=
            pyconv_kernels[2], padding=pyconv_kernels[2] // 2, stride=
            stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)
            ), dim=1)


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {'inplans': 64, 'planes': 32}]
