import torch
import torch.nn
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3,
        n_hidden4, n_hidden5, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)
        self.hidden5 = torch.nn.Linear(n_hidden4, n_hidden5)
        self.out = torch.nn.Linear(n_hidden5, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = self.out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_feature': 4, 'n_hidden1': 4, 'n_hidden2': 4,
        'n_hidden3': 4, 'n_hidden4': 4, 'n_hidden5': 4, 'n_output': 4}]
