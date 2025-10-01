import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


class XConv2d(nn.Conv2d):
    """
    X-Convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    groups : int, default 1
        Number of groups.
    expand_ratio : int, default 2
        Ratio of expansion.
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups=1,
        expand_ratio=2, **kwargs):
        super(XConv2d, self).__init__(in_channels=in_channels, out_channels
            =out_channels, kernel_size=kernel_size, groups=groups, **kwargs)
        self.expand_ratio = expand_ratio
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        grouped_in_channels = in_channels // groups
        self.mask = torch.nn.Parameter(data=torch.Tensor(out_channels,
            grouped_in_channels, *kernel_size), requires_grad=False)
        self.init_parameters()

    def init_parameters(self):
        shape = self.mask.shape
        expand_size = max(shape[1] // self.expand_ratio, 1)
        self.mask[:] = 0
        for i in range(shape[0]):
            jj = torch.randperm(shape[1], device=self.mask.device)[:expand_size
                ]
            self.mask[i, jj, :, :] = 1

    def forward(self, input):
        masked_weight = self.weight.mul(self.mask)
        return F.conv2d(input=input, weight=masked_weight, bias=self.bias,
            stride=self.stride, padding=self.padding, dilation=self.
            dilation, groups=self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
