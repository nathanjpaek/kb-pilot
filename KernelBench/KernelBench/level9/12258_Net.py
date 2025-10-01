import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 24)
        self.fc3 = nn.Linear(24, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
