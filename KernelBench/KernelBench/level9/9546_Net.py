import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (3, 8), bias=False, stride=1)
        self.fc1 = nn.Linear(25 * 4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(-1, 25 * 4)
        x = self.fc1(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 32, 32])]


def get_init_inputs():
    return [[], {}]
