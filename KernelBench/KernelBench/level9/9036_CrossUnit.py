import torch
from torch import nn
from torch.nn import functional


class CrossUnit(nn.Module):

    def __init__(self, input_dim, inner_dim, out_dim) ->None:
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, inner_dim)
        self.fc_2 = nn.Linear(inner_dim, out_dim)
        self.align = input_dim == out_dim
        if not self.align:
            self.fc_3 = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        z = self.fc_1(x).relu()
        z = self.fc_2(z)
        if not self.align:
            x = self.fc_3(x)
        return functional.relu(x + z)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'inner_dim': 4, 'out_dim': 4}]
