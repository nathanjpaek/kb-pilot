import torch
import torch.nn.functional as F


class Actor(torch.nn.Module):
    """Defines custom model
    Inherits from torch.nn.Module
    """

    def __init__(self, dim_input, dim_output):
        super(Actor, self).__init__()
        self._dim_input = dim_input
        self._dim_output = dim_output
        SIZE_H1 = 50
        SIZE_H2 = 20
        """Initialize nnet layers"""
        self._l1 = torch.nn.Linear(self._dim_input, SIZE_H1)
        self._l2 = torch.nn.Linear(SIZE_H1, SIZE_H2)
        self._l3 = torch.nn.Linear(SIZE_H2, self._dim_output)

    def forward(self, s_t):
        x = s_t
        self._l1_out = F.relu(self._l1(x))
        self._l2_out = F.relu(self._l2(self._l1_out))
        self._out = self._l3(self._l2_out)
        return self._out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_input': 4, 'dim_output': 4}]
