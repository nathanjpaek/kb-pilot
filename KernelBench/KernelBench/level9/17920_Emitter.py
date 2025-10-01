import torch
from torch import nn


class Emitter(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Emitter, self).__init__()
        self.lin_input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.relu(self.lin_input_to_hidden(x))
        h = self.dropout(h)
        h = self.relu(self.lin_hidden_to_hidden(h))
        h = self.dropout(h)
        loc = self.lin_hidden_to_loc(h)
        scale = self.softplus(self.lin_hidden_to_scale(h))
        return loc, scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}]
