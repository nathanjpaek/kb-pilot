import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class ConvNet(nn.Module):

    def __init__(self, NumChannels):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(NumChannels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 16 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 32, 32])]


def get_init_inputs():
    return [[], {'NumChannels': 4}]
