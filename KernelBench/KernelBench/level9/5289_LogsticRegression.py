import torch
import torch.nn as nn
import torch.nn.functional as F


class LogsticRegression(nn.Module):

    def __init__(self, in_dim, n_class):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim // 2)
        self.fc2 = nn.Linear(in_dim // 2, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'n_class': 4}]
