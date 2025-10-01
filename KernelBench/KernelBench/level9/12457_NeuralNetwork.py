import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, action_bound):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.action_bound = action_bound

    def forward(self, inp):
        inp = torch.tensor(inp, dtype=torch.float)
        hidden = torch.relu(self.input_layer(inp))
        hidden = torch.relu(self.hidden_layer(hidden))
        action = torch.tanh(self.output_layer(hidden))
        return self.action_bound * action


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4,
        'action_bound': 4}]
