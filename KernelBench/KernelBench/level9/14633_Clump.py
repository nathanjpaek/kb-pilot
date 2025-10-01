import torch
from torch import nn


class Clump(nn.Module):
    """Clipping input tensor."""

    def __init__(self, min_v: 'int'=-50, max_v: 'int'=50):
        """Class for preparing input for DL model with mixed data.

        Args:
            min_v: Min value.
            max_v: Max value.

        """
        super(Clump, self).__init__()
        self.min_v = min_v
        self.max_v = max_v

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = torch.clamp(x, self.min_v, self.max_v)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
