import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):

    def __init__(self, actions):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, actions)

    def forward(self, input_state):
        x = F.relu(self.conv1(input_state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 81, 81])]


def get_init_inputs():
    return [[], {'actions': 4}]
