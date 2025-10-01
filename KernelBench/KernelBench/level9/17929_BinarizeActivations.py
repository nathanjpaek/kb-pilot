import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.autograd as autograd


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


class BinarizeActivations(nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def forward(self, x):
        return BinarizeWeight.apply(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
