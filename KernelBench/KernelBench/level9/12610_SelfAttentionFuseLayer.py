import torch
from torch import nn


class SelfAttentionFuseLayer(nn.Module):

    def __init__(self, dim):
        super(SelfAttentionFuseLayer, self).__init__()
        self.W_7 = nn.Linear(dim, dim)
        self.w_8 = nn.Linear(dim, 1)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        h1 = self.W_7(hidden_states)
        h1 = self.activation(h1)
        h2 = self.w_8(h1)
        h2 = self.activation(h2)
        h2 = h2.squeeze()
        a = torch.softmax(h2, dim=1)
        a = a.unsqueeze(dim=2)
        a = a.expand_as(hidden_states)
        hidden_states = hidden_states * a
        hidden_states = torch.sum(hidden_states, dim=1)
        hidden_states = hidden_states.squeeze()
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
