import torch
import torch.nn as nn
import torch.nn.functional as F


class HSwishFunctionV2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feat):
        act = F.relu6(feat + 3).mul_(feat).div_(6)
        ctx.variables = feat
        return act

    @staticmethod
    def backward(ctx, grad_output):
        feat = ctx.variables
        grad = F.relu6(feat + 3).div_(6)
        grad.add_(torch.where(torch.eq(-3 < feat, feat < 3), torch.
            ones_like(feat).div_(6), torch.zeros_like(feat)).mul_(feat))
        grad *= grad_output
        return grad


class HSwishV2(nn.Module):

    def __init__(self):
        super(HSwishV2, self).__init__()

    def forward(self, feat):
        return HSwishFunctionV2.apply(feat)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
