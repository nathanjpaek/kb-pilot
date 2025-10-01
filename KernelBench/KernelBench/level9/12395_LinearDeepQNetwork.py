import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class LinearDeepQNetwork(nn.Module):

    def __init__(self, lr, input, n_actions):
        super(LinearDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lr': 4, 'input': 4, 'n_actions': 4}]
