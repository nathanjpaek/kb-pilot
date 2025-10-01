import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, last_layer_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding='same')
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 16, (3, 3), padding='same')
        self.last_layer_channels = 16
        self.conv4 = nn.Conv2d(16, self.last_layer_channels, (3, 3),
            padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'last_layer_channels': 1}]
