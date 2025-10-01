import torch
import torch.nn as nn
import torch.quantization


class HSigmoid(nn.Module):
    """Hard Sigmoid."""

    def __init__(self, inplace: 'bool'=True) ->None:
        """Initialize."""
        super(HSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward."""
        x = self.relu6(x + 3) / 6
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
