import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F


class CNN(nn.Module):
    """Convolutional Neural Network"""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 4)
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 48, 48])]


def get_init_inputs():
    return [[], {}]
