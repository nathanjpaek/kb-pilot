import math
import torch
import torch.nn.functional
import torch.backends.cudnn


class Conv2d(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
            dilation, groups, bias)

    def forward(self, x):
        s = self.stride
        d = self.dilation
        k = self.weight.shape[-2:]
        h, w = x.size()[-2:]
        pad_h = max((math.ceil(h / s[0]) - 1) * s[0] + (k[0] - 1) * d[0] + 
            1 - h, 0)
        pad_w = max((math.ceil(w / s[1]) - 1) * s[1] + (k[1] - 1) * d[1] + 
            1 - w, 0)
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2], value=0)
        return torch.nn.functional.conv2d(x, self.weight, self.bias, self.
            stride, (0, 0), self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
