import torch
import torch.nn as nn
import torch.nn.functional as F


class NormMLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(NormMLP, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, activations):
        return self.layer_norm(self.linear(F.relu(activations)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
