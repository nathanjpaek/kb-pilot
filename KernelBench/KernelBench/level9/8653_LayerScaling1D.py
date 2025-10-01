import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class LayerScaling1D(nn.Module):
    """Scales inputs by the second moment for the entire layer.
    .. math::

        y = \\frac{x}{\\sqrt{\\mathrm{E}[x^2] + \\epsilon}}

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)` (same shape as input)

    Examples::

        >>> ls = LayerScaling()
        >>> input = torch.randn(20, 100)
        >>> output = ls(input)

    """

    def __init__(self, eps=1e-05, **kwargs):
        super(LayerScaling1D, self).__init__()
        self.eps = eps

    def extra_repr(self):
        return f'eps={self.eps}'

    def forward(self, input):
        moment2 = torch.mean(input * input, dim=1, keepdim=True)
        return input / torch.sqrt(moment2 + self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
