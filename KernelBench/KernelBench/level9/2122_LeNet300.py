import torch
from torch import nn


class LeNet300(nn.Module):

    def __init__(self):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(784, 300, bias=True)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
