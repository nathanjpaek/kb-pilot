import torch
from torch import nn
import torch.nn.functional as F


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv2 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(2 * 2 * 50, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2 * 2 * 50)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 50, 64, 64])]


def get_init_inputs():
    return [[], {}]
