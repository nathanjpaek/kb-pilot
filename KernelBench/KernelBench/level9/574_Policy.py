import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim
import torch.autograd


class Policy(nn.Module):

    def __init__(self, learning_rate, gamma, in_dim, out_dim):
        super(Policy, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.data = []
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'learning_rate': 4, 'gamma': 4, 'in_dim': 4, 'out_dim': 4}]
