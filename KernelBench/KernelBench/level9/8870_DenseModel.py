import torch
from torch import nn


class DenseModel(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_size=150,
        activation=None):
        super(DenseModel, self).__init__()
        self.l1 = nn.Linear(input_shape, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_shape)
        self.activation = activation
        self._init_weights()

    def forward(self, input):
        x = input
        x = self.l1(x)
        x = nn.functional.tanh(x)
        x = self.l2(x)
        if self.activation:
            x = self.activation(x)
        return x

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform(layer.weight, gain=nn.init.
                    calculate_gain('relu'))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': 4, 'output_shape': 4}]
