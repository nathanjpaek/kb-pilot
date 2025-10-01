import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.linear7 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, hidden_dim)
        self.linear9 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        x3 = F.relu(self.linear7(xu))
        x3 = F.relu(self.linear8(x3))
        x3 = self.linear9(x3)
        return x1, x2, x3

    def q1(self, state, action):
        q1 = F.relu(self.linear1(torch.cat([state, action], 1)))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)
        return q1


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_actions': 4, 'hidden_dim': 4}]
