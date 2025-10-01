import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc5 = nn.Linear(64 * 21 * 21, 1000)
        self.fc6 = nn.Linear(1000, 500)
        self.fc7 = nn.Linear(500, 136)
        self.fc6_drop = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 21 * 21)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 96, 96])]


def get_init_inputs():
    return [[], {}]
