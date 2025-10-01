import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class Actor(nn.Module):

    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300,
        init_w=0.003):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nb_states': 4, 'nb_actions': 4}]
