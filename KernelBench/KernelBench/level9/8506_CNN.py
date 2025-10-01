import torch
import torch.nn.functional as F
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'num_classes': 4}]
