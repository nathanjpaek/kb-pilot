import torch
from typing import List
from torch import nn
from typing import Tuple
from typing import Union


class Add(nn.Module):
    """Add module for Kindle."""

    def __init_(self):
        """Initialize module."""
        super().__init__()

    @classmethod
    def forward(cls, x: 'Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]'
        ) ->torch.Tensor:
        """Add inputs.

        Args:
            x: list of torch tensors

        Returns:
            sum of all x's
        """
        result = x[0]
        for i in range(1, len(x)):
            result = result + x[i]
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
