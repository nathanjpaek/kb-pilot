import torch
import numpy as np
from itertools import product as product
import torch.nn.functional as F
from torch import nn
import torch.optim
import torch.utils.data


def ternary_threshold(delta: 'float'=0.7, *ws):
    """Ternary threshold find in ws."""
    assert isinstance(delta, float)
    num_params = sum_w = 0
    if not ws:
        threshold = torch.tensor(np.nan)
    else:
        for w in ws:
            num_params += w.data.numel()
            sum_w += w.abs().sum()
        threshold = delta * (sum_w / num_params)
    return threshold


class TerQuant(torch.autograd.Function):
    """TeraryNet quantization function."""

    @staticmethod
    def forward(ctx, w, threshold):
        ctx.save_for_backward(w, threshold)
        w_ter = torch.where(w > threshold, torch.tensor(1.0), torch.tensor(0.0)
            )
        w_ter = torch.where(w.abs() <= -threshold, torch.tensor(0.0), w_ter)
        w_ter = torch.where(w < -threshold, torch.tensor(-1.0), w_ter)
        return w_ter

    @staticmethod
    def backward(ctx, grad_o):
        """Back propagation using same as identity function."""
        grad_i = grad_o.clone()
        return grad_i, None


class TerConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = 0.7

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        threshold = ternary_threshold(self.delta, self.weight)
        self.weight_q = TerQuant.apply(self.weight, threshold)
        x = F.conv2d(x, self.weight_q, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
