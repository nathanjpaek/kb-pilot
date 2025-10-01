import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class ActivationClamp(nn.Module):
    """Clips the output of CN.

    .. math::
        y = clip(x, -clamp_value, clamp_value)

    Args:
        clamp_value: the value to which activations are clipped.
            Default: 5

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> ac = ActivationClamp(clamp_value)
        >>> input = torch.randn(64, 128, 32, 32)
        >>> output = ac(input)
    """

    def __init__(self, clamp_value=5, **kwargs):
        super(ActivationClamp, self).__init__()
        self.clamp_value = clamp_value

    def extra_repr(self):
        return f'clamp_value={self.clamp_value}'

    def forward(self, input):
        return torch.clamp(input, -self.clamp_value, self.clamp_value)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
