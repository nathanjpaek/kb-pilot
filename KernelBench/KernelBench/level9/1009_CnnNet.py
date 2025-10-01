import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnNet(nn.Module):

    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(15488, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 30)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2(x)
        x = F.relu(self.pool2(x))
        x = self.conv3(x)
        x = F.relu(self.pool3(x))
        x = x.view(-1, 15488)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 96, 96])]


def get_init_inputs():
    return [[], {}]
