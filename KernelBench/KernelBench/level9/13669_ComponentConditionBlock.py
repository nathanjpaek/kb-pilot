import torch
import torch.nn as nn
import torch.utils.data.distributed


class ComponentConditionBlock(nn.Module):

    def __init__(self, in_shape, n_comps):
        super().__init__()
        self.in_shape = in_shape
        self.bias = nn.Parameter(torch.zeros(n_comps, in_shape[0], 1, 1),
            requires_grad=True)

    def forward(self, x, comp_id):
        b = self.bias[comp_id]
        out = x + b
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'in_shape': [4, 4], 'n_comps': 4}]
