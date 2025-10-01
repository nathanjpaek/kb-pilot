from torch.nn import Module
import torch
import torch.optim
from torch.nn.modules.module import Module
from torch.nn.functional import *


class LocalResponseNorm(Module):

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1):
        """Applies local response normalization over an input signal composed
        of several input planes, where channels occupy the second dimension.
        Applies normalization across channels.

        .. math::

            `b_{c} = a_{c}\\left(k + \\frac{\\alpha}{n}
            \\sum_{c'=\\max(0, c-n/2)}^{\\min(N-1,c+n/2)}a_{c'}^2\\right)^{-\\beta}`

        Args:
            size: amount of neighbouring channels used for normalization
            alpha: multiplicative factor. Default: 0.0001
            beta: exponent. Default: 0.75
            k: additive factor. Default: 1

        Shape:
            - Input: :math:`(N, C, ...)`
            - Output: :math:`(N, C, ...)` (same shape as input)
        Examples::
            >>> lrn = nn.LocalResponseNorm(2)
            >>> signal_2d = autograd.Variable(torch.randn(32, 5, 24, 24))
            >>> signal_4d = autograd.Variable(torch.randn(16, 5, 7, 7, 7, 7))
            >>> output_2d = lrn(signal_2d)
            >>> output_4d = lrn(signal_4d)
        """
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return local_response_norm(input, self.size, self.alpha, self.beta,
            self.k)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.size
            ) + ', alpha=' + str(self.alpha) + ', beta=' + str(self.beta
            ) + ', k=' + str(self.k) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
