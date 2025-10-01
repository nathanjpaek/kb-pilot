import torch
from torch import nn


class Sin(nn.Module):
    """An element-wise sin activation wrapped as a nn.Module.

  Shape:
      - Input: `(N, *)` where `*` means, any number of additional dimensions
      - Output: `(N, *)`, same shape as the input

  Examples:
      >>> m = Sin()
      >>> input = torch.randn(2)
      >>> output = m(input)
  """

    def forward(self, input):
        return torch.sin(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
