import torch
import torch as t
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class Transformer(nn.Module):

    def __init__(self, input_size, num_actions, hidden_size, learning_rate=
        0.0003):
        super(Transformer, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_probs(self, state):
        return self.forward(state)

    def get_action(self, state):
        state = state.unsqueeze(0)
        probs = self.forward(Variable(state))
        sampled_action = Categorical(probs.detach())
        log_prob = t.log(probs.squeeze(0)[sampled_action])
        return sampled_action, log_prob


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_actions': 4, 'hidden_size': 4}]
