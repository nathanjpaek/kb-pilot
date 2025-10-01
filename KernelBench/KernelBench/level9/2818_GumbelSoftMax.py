import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt as sqrt
from itertools import product as product


class _GumbelSoftMax(torch.autograd.Function):
    """
    implementing the MixedOp, but carried out in a different way as DARTS

    DARTS adds all operations together, then select the maximal one to construct the final network,
    however, during the late process, more weights are assigned to the None, this is unreasonable under the
    circumstance that per operation has the unsure number of inputs.

    Thus, we modifies the original DARTS by applying way in GDAS to test.

    This class aims to compute the gradients by ourself.
    """

    @staticmethod
    def forward(ctx, weights):
        weights_norm = F.softmax(weights, dim=-1)
        ctx.saved_for_backward = weights_norm
        mask = torch.zeros_like(weights_norm)
        _, idx = weights_norm.topk(dim=-1, k=1, largest=True)
        mask[idx] = 1.0
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        gumbel_norm = ctx.saved_for_backward
        return gumbel_norm * (1 - gumbel_norm
            ) * grad_output * gumbel_norm.shape[0]


class GumbelSoftMax(nn.Module):

    def __init__(self):
        super(GumbelSoftMax, self).__init__()

    def forward(self, weights, temp_coeff=1.0):
        gumbel = -0.001 * torch.log(-torch.log(torch.rand_like(weights)))
        weights = _GumbelSoftMax.apply((weights + gumbel) / temp_coeff)
        return weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
