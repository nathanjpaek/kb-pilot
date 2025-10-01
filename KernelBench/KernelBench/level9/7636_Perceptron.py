import torch
import torch.nn as nn
import torch.nn.functional as F


class Perceptron(nn.Module):
    """Implements a 1-layer perceptron."""

    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(Perceptron, self).__init__()
        self._layer1 = nn.Linear(input_dimension, hidden_dimension)
        self._layer2 = nn.Linear(hidden_dimension, output_dimension, bias=False
            )

    def forward(self, inp):
        return F.sigmoid(self._layer2(F.relu(self._layer1(inp))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dimension': 4, 'hidden_dimension': 4,
        'output_dimension': 4}]
