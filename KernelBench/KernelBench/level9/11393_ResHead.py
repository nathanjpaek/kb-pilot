from torch.nn import Module
import torch
import torch.utils.data
import torch.nn as nn


def gap2d(_w_in):
    """Helper for building a gap2d layer."""
    return nn.AdaptiveAvgPool2d((1, 1))


def gap2d_cx(cx, _w_in):
    """Accumulates complexity of gap2d into cx = (h, w, flops, params, acts)."""
    flops, params, acts = cx['flops'], cx['params'], cx['acts']
    return {'h': 1, 'w': 1, 'flops': flops, 'params': params, 'acts': acts}


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Linear(w_in, w_out, bias=bias)


def linear_cx(cx, w_in, w_out, *, bias=False):
    """Accumulates complexity of linear into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx['h'], cx['w'], cx['flops'], cx['params'
        ], cx['acts']
    flops += w_in * w_out + (w_out if bias else 0)
    params += w_in * w_out + (w_out if bias else 0)
    acts += w_out
    return {'h': h, 'w': w, 'flops': flops, 'params': params, 'acts': acts}


class ResHead(Module):
    """ResNet head: AvgPool, 1x1."""

    def __init__(self, w_in, num_classes):
        super(ResHead, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, num_classes):
        cx = gap2d_cx(cx, w_in)
        cx = linear_cx(cx, w_in, num_classes, bias=True)
        return cx


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'w_in': 4, 'num_classes': 4}]
