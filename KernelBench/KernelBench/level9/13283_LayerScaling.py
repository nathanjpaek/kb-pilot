import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class LayerScaling(nn.Module):
    """Scales inputs by the root of the second moment for groups of channels.
    
    .. math::
        y_g = \\frac{x_g}{\\sqrt{\\mathrm{E}[x_g^2] + \\epsilon}}
    
    Args:
        group_size: size of groups
            Default: -1 (no grouping, use all channels)
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

    def __init__(self, group_size=-1, eps=1e-05, **kwargs):
        super(LayerScaling, self).__init__()
        self.eps = eps
        self.group_size = group_size

    def extra_repr(self):
        s = f'eps={self.eps}, group_size={self.group_size}'
        return s

    def forward(self, input):
        shape = input.shape
        self.group_size = shape[1
            ] if self.group_size == -1 else self.group_size
        tmp = input.view(shape[0], shape[1] // self.group_size, self.
            group_size, *shape[2:])
        moment2 = torch.mean(tmp * tmp, dim=[2, 3, 4], keepdim=True)
        out = tmp / torch.sqrt(moment2 + self.eps)
        out = out.view(shape)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
