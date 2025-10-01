import torch
import torch.nn as nn


class GatedLinear(nn.Module):

    def __init__(self, input_size, output_size):
        super(GatedLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size * 2)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        return self.glu(self.linear(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
