import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, in_dim, hidden_dim=100):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.constant(self.fc1.bias, 0.0)
        self.fc2 = nn.Linear(256, 512)
        nn.init.xavier_normal(self.fc2.weight)
        nn.init.constant(self.fc2.bias, 0.0)
        self.fc3 = nn.Linear(512, 1)
        nn.init.xavier_normal(self.fc3.weight)
        nn.init.constant(self.fc3.bias, 0.0)

    def forward(self, x, TASK=2):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        if TASK == 1 or TASK == 2:
            score = F.sigmoid(self.fc3(h2))
        else:
            score = self.fc3(h2)
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]
