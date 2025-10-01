import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.act(hidden)
        output = self.fc2(relu[:, -1, :])
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'num_layers': 1,
        'output_dim': 4}]
