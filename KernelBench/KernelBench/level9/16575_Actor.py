import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, kernel_size):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1_ = nn.Conv2d(4, 16, kernel_size=kernel_size, stride=2)
        self.conv2_ = nn.Conv2d(16, 3, kernel_size=kernel_size, stride=2)

    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.relu(self.conv1_(x))
        mu = F.tanh(self.conv2_(x))
        return mu


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
