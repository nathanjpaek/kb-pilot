import torch
from torch import nn


class MLPLayer(nn.Module):

    def __init__(self, input_size, output_size, non_linearity=torch.sigmoid):
        super().__init__()
        self.lin1 = nn.Linear(input_size, input_size // 2)
        self.lin2 = nn.Linear(input_size // 2, output_size)
        self.non_lin = non_linearity

    def forward(self, x):
        out = self.non_lin(self.lin1(x))
        return self.non_lin(self.lin2(out))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
