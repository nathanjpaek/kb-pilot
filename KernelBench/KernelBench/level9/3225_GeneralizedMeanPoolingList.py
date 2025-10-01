import torch
from abc import ABC
from torch import nn


class GeneralizedMeanPoolingList(nn.Module, ABC):
    """Applies a 2D power-average adaptive pooling over an input signal composed of
    several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size
                     will be the same as that of the input.
    """

    def __init__(self, output_size=1, eps=1e-06):
        super(GeneralizedMeanPoolingList, self).__init__()
        self.output_size = output_size
        self.eps = eps

    def forward(self, x_list):
        outs = []
        for x in x_list:
            x = x.clamp(min=self.eps)
            out = torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)
            outs.append(out)
        return torch.stack(outs, -1).mean(-1)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'output_size=' + str(self.
            output_size) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
