import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 30)
        self.fc4 = nn.Linear(30, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
