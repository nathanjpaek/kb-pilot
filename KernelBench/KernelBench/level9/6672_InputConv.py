import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_padding(kernel_size, stride, dilation):
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


class InputConv(nn.Module):

    def __init__(self, inp, outp, k=3, stride=1, dilation=1):
        super(InputConv, self).__init__()
        self.conv = nn.Conv2d(inp, outp, k, stride, padding=_get_padding(k,
            stride, dilation), dilation=dilation)

    def forward(self, x):
        return F.relu6(self.conv(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inp': 4, 'outp': 4}]
