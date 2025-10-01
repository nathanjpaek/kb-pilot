import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DuelingDeepQNetwork(nn.Module):

    def __init__(self, lr, input_dim, output_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, output_dim)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        v = self.V(flat2)
        a = self.A(flat2)
        return v, a


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lr': 4, 'input_dim': 4, 'output_dim': 4, 'fc1_dim': 4,
        'fc2_dim': 4}]
