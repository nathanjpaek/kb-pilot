import torch
import torch.nn as nn
import torch.nn.functional


class FiLM(nn.Module):

    def __init__(self, output_size, gating_size):
        super().__init__()
        self.scale = nn.Linear(gating_size, output_size[0])
        self.shift = nn.Linear(gating_size, output_size[0])

    def forward(self, x, gating):
        scale = self.scale(gating).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(gating).unsqueeze(-1).unsqueeze(-1)
        return scale * x + shift


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'output_size': [4, 4], 'gating_size': 4}]
