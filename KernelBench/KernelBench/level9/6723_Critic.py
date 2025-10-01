import torch
import torch.nn.functional as F
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.linear1 = nn.Linear(num_inputs + num_outputs, hidden_size)
        nn.init.normal_(self.linear1.weight, 0.0, 0.02)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.linear2.weight, 0.0, 0.02)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        torch.nn.init.uniform_(self.V.weight, a=-0.003, b=0.003)

    def forward(self, inputs, actions):
        x = torch.cat((inputs, actions), 1)
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.tanh(x)
        V = self.V(x)
        return V


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_inputs': 4, 'action_space': torch.
        rand([4, 4])}]
