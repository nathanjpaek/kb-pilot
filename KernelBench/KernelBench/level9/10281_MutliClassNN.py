import torch
from torch import nn


class MutliClassNN(nn.Module):

    def __init__(self, num_features, num_labels):
        super(MutliClassNN, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, 1000)
        self.fc3 = torch.nn.Linear(1000, num_labels)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4, 'num_labels': 4}]
