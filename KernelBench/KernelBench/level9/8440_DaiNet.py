import torch
import torch.nn as nn
import torch.nn.functional as F


class DaiNet(nn.Module):

    def __init__(self):
        super(DaiNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.dp = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 3)
        self.dp = nn.Dropout(0.5)
        self.fc1 = nn.Linear(24 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 32, 32])]


def get_init_inputs():
    return [[], {}]
