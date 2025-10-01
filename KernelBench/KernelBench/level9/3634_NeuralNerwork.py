import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNerwork(nn.Module):

    def __init__(self, n_features, n_targets):
        super(NeuralNerwork, self).__init__()
        self.fc1 = nn.Linear(n_features, 15)
        self.fc2 = nn.Linear(15, 10)
        self.fc3 = nn.Linear(10, n_targets)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4, 'n_targets': 4}]
