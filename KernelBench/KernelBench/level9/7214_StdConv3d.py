import torch
from torch import nn
import torch.jit
import torch.nn.functional as F
import torch.nn.functional


class StdConv3d(nn.Conv3d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False
            )
        w = (w - m) / torch.sqrt(v + 1e-05)
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.
            dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
