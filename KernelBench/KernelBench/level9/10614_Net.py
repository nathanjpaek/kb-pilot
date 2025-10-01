import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(65536, 10)
        self.maxpool = nn.AdaptiveMaxPool2d(32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = x.view(-1, 65536)
        x = F.softmax(self.fc1(x), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
