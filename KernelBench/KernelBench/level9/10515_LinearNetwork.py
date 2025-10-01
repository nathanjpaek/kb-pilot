import torch
from typing import List
import torch.nn as nn


class LinearNetwork(nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int', hidden_layers:
        'List[int]', activation: 'nn.Module'=nn.LeakyReLU):
        super(LinearNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.n_layers = len(hidden_layers)
        self.layers = nn.Sequential()
        if self.n_layers == 0:
            self.layers.add_module('single_layer', nn.Linear(input_size,
                output_size))
        else:
            for i in range(self.n_layers + 1):
                if i == 0:
                    self.layers.add_module('input_layer', nn.Linear(
                        input_size, hidden_layers[0]))
                    self.layers.add_module('input_layer_activation', self.
                        activation())
                elif i < self.n_layers:
                    self.layers.add_module(f'hidden_layer_{i}', nn.Linear(
                        hidden_layers[i - 1], hidden_layers[i]))
                    self.layers.add_module(f'input_layer_{i}_activation',
                        self.activation())
                else:
                    self.layers.add_module('output_layer', nn.Linear(
                        hidden_layers[i - 1], output_size))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        assert len(x.shape) == 2
        assert x.shape[1] == self.input_size
        return self.layers(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'hidden_layers': [4, 4]}]
