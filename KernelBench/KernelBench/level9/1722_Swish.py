import torch
import torch.nn
import torch.nn as nn


class Swish(nn.Module):
    """Applies the element-wise function:

    .. math::
        \\text{Swish}(x) = x * \\text{Sigmoid}(\\alpha * x) for constant value alpha.

    Citation: Searching for Activation Functions, Ramachandran et al., 2017, https://arxiv.org/abs/1710.05941.


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::

        >>> m = Act['swish']()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return input * torch.sigmoid(self.alpha * input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
