import torch
import torch.nn as nn
import torch.nn.functional as F


class FC2(nn.Module):
    """ Neural network definition
    """

    def __init__(self, size):
        super(FC2, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(in_features=self.size ** 2, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        self.fc3 = nn.Linear(in_features=2, out_features=4)
        self.fc4 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
