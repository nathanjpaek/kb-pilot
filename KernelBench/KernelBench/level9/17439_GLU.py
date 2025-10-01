import torch
import torch.nn as nn
import torch.nn.functional


class GLU(nn.Module):

    def __init__(self, input_size, gating_size, output_size):
        super().__init__()
        self.gate = nn.Linear(gating_size, input_size)
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x, gating):
        g = torch.sigmoid(self.gate(gating))
        return self.lin(g * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'gating_size': 4, 'output_size': 4}]
