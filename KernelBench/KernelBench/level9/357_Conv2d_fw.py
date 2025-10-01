import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class Conv2d_fw(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.
                    stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        elif self.weight.fast is not None and self.bias.fast is not None:
            out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self
                .stride, padding=self.padding)
        else:
            out = super(Conv2d_fw, self).forward(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
