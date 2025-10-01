import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.autograd as autograd
import torch.nn.functional as F


class BinarizeWeight(autograd.Function):

    @staticmethod
    def forward(ctx, scores):
        out = scores.clone()
        out[out <= 0] = -1.0
        out[out >= 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


class XnorConv(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weight(self):
        subnet = BinarizeWeight.apply(self.weight)
        return subnet

    def forward(self, x):
        w = self.get_weight()
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.
            dilation, self.groups)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
