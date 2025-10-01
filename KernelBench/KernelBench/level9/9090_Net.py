import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (5, 5), groups=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(36, 5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(-1, 36)
        x = self.fc1(x)
        x = self.relu2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
