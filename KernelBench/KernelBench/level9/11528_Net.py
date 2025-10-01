import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(36 * 36 * 32, 64)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 136)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 36 * 36 * 32)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 121, 121])]


def get_init_inputs():
    return [[], {}]
