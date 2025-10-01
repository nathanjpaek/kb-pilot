import torch
from torch import Tensor
import torch.nn


class MaxLayer(torch.nn.Module):
    """Placeholder Layer for Max operation"""

    def __init__(self):
        super(MaxLayer, self).__init__()

    def forward(self, inputs: 'Tensor'):
        return inputs.max(dim=-1)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
