import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):

    def __init__(self, cin, cout):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(cin, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 40)
        self.fc4 = nn.Linear(40, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'cin': 4, 'cout': 4}]
