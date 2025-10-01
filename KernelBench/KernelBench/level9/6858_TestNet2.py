import torch
import torch.nn as nn
import torch.nn.functional as F


class TestNet2(nn.Module):

    def __init__(self):
        super(TestNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 7, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 36, 5, padding=2)
        self.conv3 = nn.Conv2d(36, 72, 3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * 72, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 4 * 4 * 72)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
