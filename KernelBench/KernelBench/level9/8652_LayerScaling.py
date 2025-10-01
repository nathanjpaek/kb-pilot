import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class LayerScaling(nn.Module):
    """Scales inputs by the second moment for the entire layer.
    .. math::

        y = \\frac{x}{\\sqrt{\\mathrm{E}[x^2] + \\epsilon}}

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> ls = LayerScaling()
        >>> input = torch.randn(64, 128, 32, 32)
        >>> output = ls(input)

    """

    def __init__(self, eps=1e-05, **kwargs):
        super(LayerScaling, self).__init__()
        self.eps = eps

    def extra_repr(self):
        return f'eps={self.eps}'

    def forward(self, input):
        rank = input.dim()
        tmp = input.view(input.size(0), -1)
        moment2 = torch.mean(tmp * tmp, dim=1, keepdim=True)
        for _ in range(rank - 2):
            moment2 = moment2.unsqueeze(-1)
        return input / torch.sqrt(moment2 + self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
