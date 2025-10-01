import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    MLP
    """

    def __init__(self, hidden_layers, input_size, output_size, seed=1):
        """
        `hidden_layers`: list, the number of neurons for every layer;
        `input_size`: number of states;
        `output_size`: number of actions;
        `seed`: random seed.
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.Sequential()
        self.layers.add_module('Linear_inp', nn.Linear(input_size,
            hidden_layers[0]))
        self.layers.add_module('Act_inp', nn.ReLU())
        for i in range(1, len(hidden_layers)):
            self.layers.add_module('Linear_{}'.format(i), nn.Linear(
                hidden_layers[i - 1], hidden_layers[i]))
            self.layers.add_module('Act_{}'.format(i), nn.ReLU())
        self.layers.add_module('Linear_out', nn.Linear(hidden_layers[-1],
            output_size))
        self.layers.add_module('Act_out', nn.Softmax(dim=1))

    def forward(self, input_seq):
        """
        `input_seq`: states, torch.FloatTensor.
        """
        return self.layers(input_seq)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_layers': [4, 4], 'input_size': 4, 'output_size': 4}]
