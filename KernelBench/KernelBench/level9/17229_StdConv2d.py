import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms.functional as F
from torch.nn import functional as F


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        s = w.std(dim=[1, 2, 3], keepdim=True)
        m = w.mean(dim=[1, 2, 3], keepdim=True)
        v = s * s
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.
            dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
