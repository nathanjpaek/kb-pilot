import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        self.dropout(x)
        x = F.relu(self.fc2(x))
        self.dropout(x)
        x = F.relu(self.fc3(x))
        self.dropout(x)
        x = self.fc4(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
