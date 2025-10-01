import torch
from torch import Tensor
from torch import nn


class WrapperKLDiv(nn.Module):
    """Wrapper for KL-Divergence for easy argument passing."""

    def __init__(self, reduction: 'str'='mean') ->None:
        """Constructor.

        Args:
            reduction (str, optional): One of 'none','batchmean','sum', 'mean'.
                Defaults to 'mean'.
        """
        super(WrapperKLDiv, self).__init__()
        self.reduction = reduction

    def forward(self, set1: 'Tensor', set2: 'Tensor') ->Tensor:
        """Computes the KL-Divergence.

        Args:
            set1 (Tensor): Input tensor of arbitrary shape.
            set2 (Tensor): Tensor of the same shape as input.

        Returns:
            Tensor: Scalar by default. if reduction = 'none', then same
                shape as input.
        """
        return nn.functional.kl_div(set1, set2, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
