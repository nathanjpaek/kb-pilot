import torch
import numpy as np
from abc import ABC
import torch.nn.functional as F
from torch import nn
from torch.nn import init


def fan_in_init(tensor):
    fan_in = tensor.size(1)
    v = 1.0 / np.sqrt(fan_in)
    init.uniform_(tensor, -v, v)


class Architecture(nn.Module, ABC):

    def __init__(self):
        super().__init__()


class CriticArchitecture(Architecture):

    def __init__(self, input_size, hidden_layers, output_size,
        output_activation):
        """
Initialize a Critic for low dimensional environment.
    num_feature: number of features of input.

"""
        super().__init__()
        self._input_size = input_size
        self._hidden_layers = hidden_layers
        self._output_size = output_size
        self.fc1 = nn.Linear(self._input_size[0], self._hidden_layers[0])
        fan_in_init(self.fc1.weight)
        self.fc2 = nn.Linear(self._hidden_layers[0] + self._output_size[0],
            self._hidden_layers[1])
        fan_in_init(self.fc2.weight)
        self.head = nn.Linear(self._hidden_layers[1], 1)
        init.uniform_(self.head.weight, -0.003, 0.003)
        init.uniform_(self.head.bias, -0.003, 0.003)

    def forward(self, states, actions):
        x = F.relu(self.fc1(states))
        x = torch.cat((x, actions), 1)
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': [4, 4], 'hidden_layers': [4, 4],
        'output_size': [4, 4], 'output_activation': 4}]
