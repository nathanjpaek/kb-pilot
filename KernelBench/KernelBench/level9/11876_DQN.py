import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, state_dim, nb_actions, hidden1=50, hidden2=50):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return self.fc3(out.view(out.size(0), -1))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'nb_actions': 4}]
