import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class TripletLogExpLoss(nn.Module):
    """Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.

    The distance is described in detail in the paper `Improving Pairwise Ranking for Multi-Label
    Image Classification`_ by Y. Li et al.

    .. math::
        L(a, p, n) = log \\left( 1 + exp(d(a_i, p_i) - d(a_i, n_i) \\right)


    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    >>> triplet_loss = nn.TripletLogExpLoss(p=2)
    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> input3 = autograd.Variable(torch.randn(100, 128))
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, p=2, eps=1e-06, swap=False):
        super(TripletLogExpLoss, self).__init__()
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        assert anchor.size() == positive.size(
            ), 'Input sizes between positive and negative must be equal.'
        assert anchor.size() == negative.size(
            ), 'Input sizes between anchor and negative must be equal.'
        assert positive.size() == negative.size(
            ), 'Input sizes between positive and negative must be equal.'
        assert anchor.dim() == 2, 'Input must be a 2D matrix.'
        d_p = F.pairwise_distance(anchor, positive, self.p, self.eps)
        d_n = F.pairwise_distance(anchor, negative, self.p, self.eps)
        if self.swap:
            d_s = F.pairwise_distance(positive, negative, self.p, self.eps)
            d_n = torch.min(d_n, d_s)
        dist = torch.log(1 + torch.exp(d_p - d_n))
        loss = torch.mean(dist)
        return loss

    def eval_func(self, dp, dn):
        return np.log(1 + np.exp(dp - dn))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
