import torch
import torch.nn as nn


class myNet(nn.Module):

    def __init__(self, in_features, num_classes=10):
        super(myNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
